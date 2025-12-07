using Trixi
using SparseConnectivityTracer # For obtaining the Jacobian sparsity pattern
using SparseMatrixColorings # For obtaining the coloring vector
using OrdinaryDiffEqSDIRK, ADTypes

###############################################################################
### equation, solver, mesh ###

advection_velocity = (0.2, -0.7)
equation = LinearScalarAdvectionEquation2D(advection_velocity)

solver = DGSEM(polydeg = 3, surface_flux = flux_godunov)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000)

###############################################################################
### semidiscretization for sparsity detection ###

jac_detector = TracerSparsityDetector()
# We need to construct the semidiscretization with the correct
# sparsity-detection ready datatype, which is retrieved here
jac_eltype = jacobian_eltype(real(solver), jac_detector)

# Semidiscretization for sparsity pattern detection
semi_jac_type = SemidiscretizationHyperbolic(mesh, equation,
                                             initial_condition_convergence_test, solver,
                                             uEltype = jac_eltype) # Need to supply Jacobian element type

tspan = (0.0, 1.0) # Re-used for wrapping `rhs` below

# Call `semidiscretize` to create the ODE problem to have access to the
# initial condition based on which the sparsity pattern is computed
ode_jac_type = semidiscretize(semi_jac_type, tspan)
u0_ode = ode_jac_type.u0
du_ode = similar(u0_ode)

###############################################################################
### Compute the Jacobian sparsity pattern ###

# Wrap the `Trixi.rhs!` function to match the signature `f!(du, u)`, see
# https://adrianhill.de/SparseConnectivityTracer.jl/stable/user/api/#ADTypes.jacobian_sparsity
rhs_wrapped! = (du_ode, u0_ode) -> Trixi.rhs!(du_ode, u0_ode, semi_jac_type, tspan[1])

jac_prototype = jacobian_sparsity(rhs_wrapped!, du_ode, u0_ode, jac_detector)

# For most efficient solving we also want the coloring vector

coloring_prob = ColoringProblem(; structure = :nonsymmetric, partition = :column)
coloring_alg = GreedyColoringAlgorithm(; decompression = :direct)
coloring_result = coloring(jac_prototype, coloring_prob, coloring_alg)
coloring_vec = column_colors(coloring_result)

###############################################################################
### sparsity-aware semidiscretization and ode ###

# Semidiscretization for actual simulation. `eEltype` is here retrieved from `solver`
semi_float_type = SemidiscretizationHyperbolic(mesh, equation,
                                               initial_condition_convergence_test,
                                               solver)

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_jac_sparse = semidiscretize(semi_float_type, tspan,
                                jac_prototype = jac_prototype,
                                colorvec = coloring_vec)
# using "dense" `ode = semidiscretize(semi_float_type, tspan)` is 10-15 times slower!

###############################################################################
### callbacks & solve ###

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi_float_type, interval = 10)
save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

# Note: No `stepsize_callback` due to (implicit) solver with adaptive timestep control
callbacks = CallbackSet(summary_callback, analysis_callback, save_restart)

###############################################################################
### solve the ODE problem ###

sol = solve(ode_jac_sparse,
            # Default `AutoForwardDiff()` is not yet working, see
            # https://github.com/trixi-framework/Trixi.jl/issues/2369
            TRBDF2(; autodiff = AutoFiniteDiff());
            dt = 0.1, save_everystep = false, callback = callbacks);
