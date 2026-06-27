using OrdinaryDiffEqLowOrderRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

equations = LinearScalarAdvectionEquation2D(1.0, 1.0)

# The initial condition can also be used as analytical solution, e.g.,
# for convergence tests.
function initial_condition(x, t, equations::LinearScalarAdvectionEquation2D)
    a = equations.advection_velocity
    return SVector(sinpi(x[1] - a[1] * t) * sinpi(x[2] - a[2] * t))
end

# Create DG solver with polynomial degree = 0, i.e., a first order finite volume solver,
# with (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 0, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                periodicity = true)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_periodic)
###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 10.0
ode = semidiscretize(semi, (0.0, 10.0));
# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.9)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, Euler();
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
