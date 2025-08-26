using Trixi
using SparseConnectivityTracer # For obtaining the Jacobian sparsity pattern
using SparseMatrixColorings # For obtaining the coloring vector
using OrdinaryDiffEqSDIRK, ADTypes

###############################################################################
### set up sparsity detection ###

float_type = Float64 # Datatype for the actual simulation

jac_detector = TracerSparsityDetector()
# We need to construct the semidiscretization with the correct
# Sparsity-detection ready datatype, which is retrieved here
jac_eltype = jacobian_eltype(float_type, jac_detector)

# In the Trixi implementation, we overload the sqrt function to first check if the argument 
# is < 0 and then return NaN instead of an error.
# To turn this behaviour off for the datatype used in sparsity detection,
# we switch back to the Base implementation here which does not contain an if-clause.
Trixi.sqrt(x::jac_eltype) = Base.sqrt(x)

###############################################################################
### equations and solver ###

equations = CompressibleEulerEquations2D(1.4)

# For sparsity detection we can only use `flux_lax_friedrichs` at the moment since this is 
# `if`-clause free (although it contains `min` and `max` operations.
# The sparsity pattern, however, should be the same for other (two-point) fluxes as well.
surface_flux = flux_lax_friedrichs

solver = DGSEM(polydeg = 3, surface_flux = surface_flux, RealT = float_type)

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
### semidiscretization ###

initial_condition = initial_condition_convergence_test

# Semidiscretization for sparsity pattern detection
semi_jac_type = SemidiscretizationHyperbolic(mesh, equations,
                                             initial_condition_convergence_test,
                                             solver,
                                             source_terms = source_terms_convergence_test,
                                             uEltype = jac_eltype) # Need to supply Jacobian element type

t0 = 0.0 # Re-used for wrapping `rhs` below
t_end = 5.0
t_span = (t0, t_end)

# Call `semidiscretize` to create the ODE problem to have access to the
# initial condition based on which the sparsity pattern is computed
ode_jac_type = semidiscretize(semi_jac_type, t_span)
u0_ode = ode_jac_type.u0
du_ode = similar(u0_ode)

###############################################################################
### Compute the Jacobian sparsity pattern ###

# Wrap the `Trixi.rhs!` function to match the signature `f!(du, u)`, see
# https://adrianhill.de/SparseConnectivityTracer.jl/stable/user/api/#ADTypes.jacobian_sparsity
rhs_wrapped! = (du_ode, u0_ode) -> Trixi.rhs!(du_ode, u0_ode, semi_jac_type, t0)

jac_prototype = jacobian_sparsity(rhs_wrapped!, du_ode, u0_ode, jac_detector)

# For most efficient solving we also want the coloring vector

coloring_prob = ColoringProblem(; structure = :nonsymmetric, partition = :column)
coloring_alg = GreedyColoringAlgorithm(; decompression = :direct)
coloring_result = coloring(jac_prototype, coloring_prob, coloring_alg)
coloring_vec = column_colors(coloring_result)

###############################################################################
### sparsity-aware semidiscretization and ode ###

# Semidiscretization for actual simulation
semi_float_type = SemidiscretizationHyperbolic(mesh, equations,
                                               initial_condition_convergence_test,
                                               solver,
                                               source_terms = source_terms_convergence_test,
                                               uEltype = float_type) # Not necessary, also retrieved from `solver`

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_jac_sparse = semidiscretize(semi_float_type, t_span,
                                jac_prototype = jac_prototype,
                                colorvec = coloring_vec)

###############################################################################
### callbacks & solve ###

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi_float_type, interval = 50)
alive_callback = AliveCallback(alive_interval = 3)

# Note: No `stepsize_callback` due to implicit solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

sol = solve(ode_jac_sparse, # using `ode` is essentially infeasible, even single step takes ages!
            # Default `AutoForwardDiff()` is not yet working,
            # probably related to https://docs.sciml.ai/DiffEqDocs/stable/basics/faq/#Autodifferentiation-and-Dual-Numbers
            Kvaerno4(; autodiff = AutoFiniteDiff());
            dt = 0.05, save_everystep = false, callback = callbacks);
