using Trixi
using OrdinaryDiffEq
using SparseDiffTools, Symbolics

using .TrixiSparseDiffToolsExt

###############################################################################
### semidiscretization of the linear advection equation ###

advection_velocities = (1.0, 1.1)
equations = LinearScalarAdvectionEquation2D(advection_velocities)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
d = 3
# `RealT = Real` requires fewer overloads than the more explicit `RealT = Num`
# solver_real used for computing the Jacobian
solver_real = DGSEM(polydeg = d, surface_flux = flux_lax_friedrichs, RealT = Real)
# solver_float used for solving using the Jacobian
solver_float = DGSEM(polydeg = d, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
RefinementLevel = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = RefinementLevel,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
# semi_real used for computing the Jacobian
semi_real = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver_real)
# semi_float used for solving using the Jacobian
semi_float = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver_float)

# 2^RefinementLevel = 16 elements, (d+1) local polynomial coefficients per element
N = Int(2^RefinementLevel * (d+1)) #64
u0_ode = zeros(N*N)
du_ode = 80 * ones(N*N) # initialize with something
t0 = 0.0
tSpan = (t0, t0 + 10.0)

###############################################################################
### Compute the Jacobian with SparseDiffTools ###

# Create a function with two parameters:du_ode and u0_ode
# to fulfill the requirments of an in_place function in SparseDiffTools
rhs = (du_ode,u0_ode)->Trixi.rhs!(du_ode, u0_ode, semi_real, t0)

#From the example to detect the pattern and choose how to do the AutoDiff automatically
sd = SymbolicsSparsityDetection()
adtype = AutoSparseFiniteDiff()

#From the example provided in SparseDiffTools. cache will reduce calculation time when Jacobian will be calculated multiple times
sparse_cache = sparse_jacobian_cache(adtype, sd, rhs, du_ode, u0_ode)

###############################################################################
#  callback functions during the time integration

ode = semidiscretize(semi_float, tSpan, sparse_cache.jac_prototype, sparse_cache.coloring.colorvec)

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi_float, interval = 100)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback)

###############################################################################
# Run the simulation using ImplicitEuler method

sol = solve(ode, ImplicitEuler(; autodiff = AutoFiniteDiff());
            dt = 1.0, save_everystep = false, callback = callbacks);
