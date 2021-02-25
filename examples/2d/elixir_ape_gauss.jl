using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the acoustic perturbation equations

v_mean = (0.25, 0.25)
rho_mean = 1.0
c_mean = 1.0
equations = AcousticPerturbationEquations2D(v_mean, rho_mean, c_mean)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(3, flux_lax_friedrichs)

coordinates_min = (-1, -1) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1,  1) # maximum coordinates (max(x), max(y))

# Create a uniformely refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_gauss, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# Enable in-situ visualization with a new plot generated every 20 time steps
# and additional plotting options passed as keyword arguments
visualization = VisualizationCallback(interval=20, clims=(-0.5,1), variable_names=["p_prime"])

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.5)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, visualization, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
