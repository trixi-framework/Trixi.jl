using Trixi
using SparseConnectivityTracer # For obtaining the Jacobian sparsity pattern
using SparseMatrixColorings # For obtaining the coloring vector
using OrdinaryDiffEqSDIRK, ADTypes

###############################################################################
### solver and equations ###

# For sparsity detection we can only use `flux_lax_friedrichs` at the moment since this is 
# `if`-clause free (although it contains `min` and `max` operations).
# The sparsity pattern, however, should be the same for other (two-point) fluxes as well.
surface_flux = flux_lax_friedrichs
solver = DGSEM(polydeg = 3, surface_flux = surface_flux)

equations = CompressibleEulerEquations2D(1.4)

###############################################################################
### mesh ###

# Mapping as described in https://arxiv.org/abs/2012.12040,
# reduced to 2D on [0, 2]^2 instead of [0, 3]^3
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,2]
    xi = xi_ + 1
    eta = eta_ + 1

    y = eta + 1 / 4 * (cos(pi * (xi - 1)) *
                       cos(0.5 * pi * (eta - 1)))

    x = xi + 1 / 4 * (cos(0.5 * pi * (xi - 1)) *
                      cos(2 * pi * (y - 1)))

    return SVector(x, y)
end
cells_per_dimension = (16, 16)
mesh = StructuredMesh(cells_per_dimension, mapping)

###############################################################################
### semidiscretization for sparsity detection ###

jac_detector = TracerSparsityDetector()
# We need to construct the semidiscretization with the correct
# sparsity-detection ready datatype, which is retrieved here
jac_eltype = jacobian_eltype(real(solver), jac_detector)

# Semidiscretization for sparsity pattern detection
# Must be called 'semi' in order for the convergence test to run successfully
semi_jac_type = SemidiscretizationHyperbolic(mesh, equations,
                                             initial_condition_convergence_test,
                                             solver,
                                             source_terms = source_terms_convergence_test,
                                             uEltype = jac_eltype) # Need to supply Jacobian element type

tspan = (0.0, 5.0) # Re-used for wrapping `rhs` below

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

# Semidiscretization for actual simulation
semi = SemidiscretizationHyperbolic(mesh, equations,
                                               initial_condition_convergence_test,
                                               solver,
                                               source_terms = source_terms_convergence_test)

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_jac_sparse = semidiscretize(semi, tspan,
                                jac_prototype = jac_prototype,
                                colorvec = coloring_vec)
# using "dense" `ode = semidiscretize(semi, tspan)`
# is essentially infeasible, even single step takes ages!

###############################################################################
### callbacks & solve ###

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 50)
alive_callback = AliveCallback(alive_interval = 3)

# Note: No `stepsize_callback` due to implicit solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

sol = solve(ode_jac_sparse,
            # Default `AutoForwardDiff()` is not yet working, see
            # https://github.com/trixi-framework/Trixi.jl/issues/2369
            Kvaerno4(; autodiff = AutoFiniteDiff());
            dt = 0.05, save_everystep = false, callback = callbacks);
