using Trixi
using OrdinaryDiffEqSDIRK

# Functionality for automatic sparsity detection
using SparseDiffTools, Symbolics

# We need to avoid if-clauses to be able to use `Num` type from Symbolics.
# In the Trixi implementation, we overload the sqrt function to first check if the argument 
# is < 0 and then return NaN instead of an error.
# To turn off this behaviour, we switch back to the Base implementation here
Trixi.set_sqrt_type!("sqrt_Base")

include("../../ext/TrixiSparseDiffToolsExt.jl")
using .TrixiSparseDiffToolsExt

###############################################################################
### semidiscretization of the linear advection equation ###

equations = CompressibleEulerEquations1D(1.4)

# Can for sparsity detection only use `flux_lax_friedrichs` at the moment since this is 
# `if`-clause free

# `RealT = Real` requires fewer overloads than the more explicit `RealT = Num`
# `solver_real` is used for computing the Jacobian sparsity pattern
solver_real = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs, RealT = Real)
# `solver_float` is  used for the subsequent simulation
solver_float = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 10_000)

initial_condition = initial_condition_convergence_test

# `semi_real` is used for computing the Jacobian sparsity pattern
semi_real = SemidiscretizationHyperbolic(mesh, equations,
                                         initial_condition_convergence_test,
                                         solver_real,
                                         source_terms = source_terms_convergence_test)
# `semi_float` is  used for the subsequent simulation
semi_float = SemidiscretizationHyperbolic(mesh, equations,
                                          initial_condition_convergence_test,
                                          solver_float,
                                          source_terms = source_terms_convergence_test)

t0 = 0.0
t_end = 2.0
t_span = (t0, t_end)

# Call `semidiscretize` to create the ODE problem to have access to the initial condition.
# For the linear example considered here one could also use an arbitrary vector for the initial condition.
ode = semidiscretize(semi_float, t_span)
u0_ode = ode.u0
du_ode = similar(u0_ode)

###############################################################################
### Compute the Jacobian with SparseDiffTools ###

# Create a function with two parameters: `du_ode` and `u0_ode`
# to fulfill the requirments of an in_place function in SparseDiffTools
# (see example function `f` from https://docs.sciml.ai/SparseDiffTools/dev/#Example)
rhs = (du_ode, u0_ode) -> Trixi.rhs!(du_ode, u0_ode, semi_real, t0)

# Taken from example linked above to detect the pattern and choose how to do the AutoDiff automatically
sd = SymbolicsSparsityDetection()
ad_type = AutoForwardDiff()
sparse_adtype = AutoSparse(ad_type)

# `sparse_cache` will reduce calculation time when Jacobian is calculated multiple times
sparse_cache = sparse_jacobian_cache(sparse_adtype, sd, rhs, du_ode, u0_ode)

###############################################################################

# Supply Jacobian prototype and coloring vector to the semidiscretization
ode_float_detection = semidiscretize(semi_float, t_span,
                                     sparse_cache.jac_prototype,
                                     sparse_cache.coloring.colorvec)

analysis_callback = AnalysisCallback(semi_float, interval = 100)
summary_callback = SummaryCallback()

# Note: No `stepsize_callback` due to implicit solver
callbacks = CallbackSet(analysis_callback, summary_callback)

###############################################################################
# Run the simulation using ImplicitEuler method

sol = solve(ode, Kvaerno4(; autodiff = AutoFiniteDiff()); # `AutoForwardDiff()` is not yet working
            adaptive = false,
            dt = 0.5, save_everystep = false, callback = callbacks);
