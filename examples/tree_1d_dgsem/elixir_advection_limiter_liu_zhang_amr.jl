using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Step function initial condition which is 1 on [-0.5, 0.5] and zero elsewhere
function initial_condition_heaviside_step(x, t, equations::LinearScalarAdvectionEquation1D)
    x1 = x[1]
    u = abs(x1) < 0.5f0 ? one(x1) : zero(x1)
    return SVector(u)
end
initial_condition = initial_condition_heaviside_step

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate

# Create a mesh with periodic boundaries and adaptive refinement
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                periodicity = true)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with given time span
ode = semidiscretize(semi, (0.0, 2.0))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi,
                                      IndicatorMax(semi, variable = first),
                                      base_level = 4,
                                      med_level = 5, med_threshold = 0.1,
                                      max_level = 6, max_threshold = 0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
# We use a large CFL number here, which also causes Zhang-Shu limiting to fail.
stepsize_callback = StepsizeCallback(cfl = 1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        amr_callback, stepsize_callback)

###############################################################################
# run the simulation

# The Zhang-Shu limiter does not work by itself. Using the Liu-Zhang limiter
# resolves this by redistributing cell averages to satisfy positivity constraints.
# Note the threshold is significantly larger than implied by the initial condition
# to stress-test the limiter.
# 
# For scalar equations, the projection to the admissible set assumes that 
# `variables = (first,)` for the Liu-Zhang limiter.
local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1e-3,),
                                                     variables = (first,))
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

sol = solve(ode,
            RDPK3SpFSAL35(; stage_limiter! = global_limiter!,
                          step_limiter! = global_limiter!);
            adaptive = false,
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
