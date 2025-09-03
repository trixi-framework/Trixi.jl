using Trixi
using SparseConnectivityTracer # For obtaining the Jacobian sparsity pattern
using SparseMatrixColorings # For obtaining the coloring vector
using OrdinaryDiffEqSDIRK, ADTypes

###############################################################################
# semidiscretization of the linear advection-diffusion equation

advection_velocity = (1.5, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = true,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

# Define initial condition
function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation2D)
    # Store translated coordinate for easy use of exact solution
    RealT = eltype(x)
    x_trans = x - equation.advection_velocity * t

    nu = diffusivity()
    c = 1
    A = 0.5f0
    L = 2
    f = 1.0f0 / L
    omega = 2 * convert(RealT, pi) * f
    scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

###############################################################################
### semidiscretization for sparsity detection ###

jac_detector = TracerSparsityDetector()
# We need to construct the semidiscretization with the correct
# sparsity-detection ready datatype, which is retrieved here
jac_eltype = jacobian_eltype(real(solver), jac_detector)

# A semidiscretization collects data structures and functions for the spatial discretization
semi_jac_type = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver,
                                             uEltype = jac_eltype;
                                             solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

tspan = (0.0, 1.5) # Re-used for wrapping `rhs_parabolic!` below

# Call `semidiscretize` to create the ODE problem to have access to the
# initial condition based on which the sparsity pattern is computed
ode_jac_type = semidiscretize(semi_jac_type, tspan)
u0_ode = ode_jac_type.u0
du_ode = similar(u0_ode)

###############################################################################
### Compute the Jacobian sparsity pattern ###

# Only the parabolic part of the `SplitODEProblem` is treated implicitly so we only need the parabolic Jacobian
# https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/#SciMLBase.SplitFunction

# Wrap the `Trixi.rhs_parabolic!` function to match the signature `f!(du, u)`, see
# https://adrianhill.de/SparseConnectivityTracer.jl/stable/user/api/#ADTypes.jacobian_sparsity
rhs_parabolic_wrapped! = (du_ode, u0_ode) -> Trixi.rhs_parabolic!(du_ode, u0_ode, semi_jac_type, tspan[1])

jac_prototype_parabolic = jacobian_sparsity(rhs_parabolic_wrapped!, du_ode, u0_ode, jac_detector)

# For most efficient solving we also want the coloring vector

coloring_prob = ColoringProblem(; structure = :nonsymmetric, partition = :column)
coloring_alg = GreedyColoringAlgorithm(; decompression = :direct)
coloring_result = coloring(jac_prototype_parabolic, coloring_prob, coloring_alg)
coloring_vec_parabolic = column_colors(coloring_result)

###############################################################################
### sparsity-aware semidiscretization and ode ###

# Semidiscretization for actual simulation. `eEltype` is here retrieved from `solver`
semi_float_type = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = ViscousFormulationBassiRebay1(),
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_jac_sparse = semidiscretize(semi_float_type, tspan,
                                jac_prototype_parabolic = jac_prototype_parabolic,
                                colorvec_parabolic = coloring_vec_parabolic)
# using "dense" `ode = semidiscretize(semi_float_type, tspan)` is 4-6 times slower!

###############################################################################
### callbacks  ###

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi_float_type, interval = analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

# Note: No `stepsize_callback` due to (implicit) solver with adaptive timestep control
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_restart)

###############################################################################
### solve the ODE problem ###

time_int_tol = 1.0e-9
time_abs_tol = 1.0e-9

sol = solve(ode_jac_sparse, TRBDF2(; autodiff = AutoFiniteDiff());
            dt = 0.1, save_everystep = false,
            abstol = time_abs_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)