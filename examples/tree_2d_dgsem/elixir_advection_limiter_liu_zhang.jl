using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (1.0, 0.5)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Step function initial condition which is 1 on [-0.5, 0.5] and zero elsewhere
function initial_condition_heaviside_step(x, t, equations::LinearScalarAdvectionEquation2D)
    u = abs(x[1]) < 0.5 && abs(x[2]) < 0.5 ? 1.0 : 0.0
    return SVector(u)
end
initial_condition = initial_condition_heaviside_step

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinate
coordinates_max = (1.0, 1.0) # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                periodicity = true)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 4.0))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
# We use a large CFL number here, which causes Zhang-Shu limiting by itself to fail.
stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1e-1,),
                                                     variables = ((u, equations) -> u[1],))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

sol = solve(ode,
            CarpenterKennedy2N54(; stage_limiter! = global_limiter!,
                                 step_limiter! = global_limiter!,
                                 williamson_condition = false);
            adaptive = false, dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
