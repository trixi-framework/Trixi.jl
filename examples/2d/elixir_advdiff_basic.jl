
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (0.2, 0.6)
nu = 1.2e-2
equations = LinearAdvectionDiffusionEquation2D(advectionvelocity, nu)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
polydeg = 3 # 4
solver_hyperbolic = DGSEM(polydeg, flux_lax_friedrichs)

coordinates_min = (-1, -1) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1,  1) # maximum coordinates (max(x), max(y))
refinement_patches = (
  (type="box", coordinates_min=(0.0, -1.0), coordinates_max=(1.0, 1.0)),
)

# Create a uniformely refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                # refinement_patches=refinement_patches,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
initial_condition = initial_condition_convergence_test
semi_hyperbolic = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_hyperbolic)

solver_parabolic = DGSEM(polydeg, flux_central)
semi_parabolic = SemidiscretizationParabolicAuxVars(mesh, equations, initial_condition, solver_parabolic)

semi = SemidiscretizationHyperbolicParabolic(semi_hyperbolic, semi_parabolic)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
# stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
# callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1e-3, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
