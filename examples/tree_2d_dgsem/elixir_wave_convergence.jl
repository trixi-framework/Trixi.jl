using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the wave equations

equations = WaveEquations2D(2 * sqrt(2 / 5))

# initial condition for a standing wave
initial_condition = function (x, t, equations::WaveEquations2D)
    c = equations.c
    p = cospi(3x[1] / 2) * cospi(x[2] / 2) * cospi(sqrt(5 / 2) * c * t)
    vx = 3 / sqrt(10) * sinpi(3x[1] / 2) * cospi(x[2] / 2) * sinpi(sqrt(5 / 2) * c * t)
    vy = 1 / sqrt(10) * cospi(3x[1] / 2) * sinpi(x[2] / 2) * sinpi(sqrt(5 / 2) * c * t)
    return SVector(p, vx, vy)
end

# corresponding boundary condition for the standing wave
boundary_condition = function (u_inner, orientation, direction, x, t,
                               surface_flux_function,
                               equations::WaveEquations2D)
    u_boundary = initial_condition(x, t, equations)
    # Calculate boundary flux
    if direction in (2, 4)  # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
    end
    return flux
end

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000, periodicity = false)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

analysis_interval = 100

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.8)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
