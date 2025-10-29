using Trixi
using SparseConnectivityTracer # For obtaining the Jacobian sparsity pattern
using SparseMatrixColorings # For obtaining the coloring vector
using OrdinaryDiffEqBDF, ADTypes

###############################################################################
# semidiscretization of the linear advection-diffusion equation

advection_velocity = 1.5
equations_hyperbolic = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations_hyperbolic)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation1D)
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

###############################################################################
### semidiscretization for sparsity detection ###

jac_detector = TracerSparsityDetector()
# We need to construct the semidiscretization with the correct
# sparsity-detection ready datatype, which is retrieved here
jac_eltype = jacobian_eltype(real(solver), jac_detector)

semi_jac_type = SemidiscretizationHyperbolicParabolic(mesh,
                                                      (equations_hyperbolic,
                                                       equations_parabolic),
                                                      initial_condition, solver,
                                                      uEltype = jac_eltype)

tspan = (0.0, 1.5) # Re-used for wrapping `rhs_parabolic!` below

# Call `semidiscretize` to create the ODE problem to have access to the
# initial condition based on which the sparsity pattern is computed
ode_jac_type = semidiscretize(semi_jac_type, tspan)
u0_ode = ode_jac_type.u0
du_ode = similar(u0_ode)

###############################################################################
### Compute the Jacobian sparsity pattern ###

# Only the parabolic part of the `SplitODEProblem` is treated implicitly so we only need the parabolic Jacobian, see
# https://docs.sciml.ai/DiffEqDocs/stable/types/split_ode_types/#SciMLBase.SplitFunction
# Wrap the `Trixi.rhs_parabolic!` function to match the signature `f!(du, u)`, see
# https://adrianhill.de/SparseConnectivityTracer.jl/stable/user/api/#ADTypes.jacobian_sparsity
rhs_parabolic_wrapped! = (du_ode, u0_ode) -> Trixi.rhs_parabolic!(du_ode, u0_ode,
                                                                  semi_jac_type, tspan[1])

jac_prototype_parabolic = jacobian_sparsity(rhs_parabolic_wrapped!, du_ode, u0_ode,
                                            jac_detector)

# For most efficient solving we also want the coloring vector

# We choose `nonsymmetric` `structure` because we're computing a Jacobian, which
# is for the Upwind-alike discretization of the advection term nonsymmmetric
# We arbitrarily choose a column-based `partitioning`. This means that we will color
# structurally orthogonal columns the same. `row` partitioning would be equally valid here
coloring_prob = ColoringProblem(; structure = :nonsymmetric, partition = :column)
# The `decompression` arg specifies our evaluation scheme. The `direct` method requires solving
# a diagonal system, whereas the `substitution` method solves a triangular system of equations
coloring_alg = GreedyColoringAlgorithm(; decompression = :direct)
coloring_result = coloring(jac_prototype_parabolic, coloring_prob, coloring_alg)
coloring_vec_parabolic = column_colors(coloring_result)

###############################################################################
### sparsity-aware semidiscretization and ODE ###

# Semidiscretization for actual simulation. `uEltype` is here retrieved from `solver`
semi_float_type = SemidiscretizationHyperbolicParabolic(mesh,
                                                        (equations_hyperbolic,
                                                         equations_parabolic),
                                                        initial_condition, solver)

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_jac_sparse = semidiscretize(semi_float_type, tspan,
                                jac_prototype_parabolic = jac_prototype_parabolic,
                                colorvec_parabolic = coloring_vec_parabolic)
# using "dense" `ode = semidiscretize(semi_float_type, tspan)` is 4-6 times slower!

###############################################################################
### callbacks  ###

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi_float_type, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

# Note: No `stepsize_callback` due to (implicit) solver with adaptive timestep control
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_restart)

###############################################################################
### solve the ODE problem ###

sol = solve(ode_jac_sparse, SBDF2(; autodiff = AutoFiniteDiff());
            dt = 0.01, save_everystep = false,
            abstol = 1e-9, reltol = 1e-9,
            ode_default_options()..., callback = callbacks)
