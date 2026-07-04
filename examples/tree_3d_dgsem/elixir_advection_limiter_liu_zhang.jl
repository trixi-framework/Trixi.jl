using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (1.0, 0.5, 0.25)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

# Step function initial condition which is 1 on [-0.5, 0.5]^3 and zero elsewhere
function initial_condition_heaviside_step(x, t, equations::LinearScalarAdvectionEquation3D)
    x1, x2, x3 = x
    u = abs(x1) < 0.5f0 && abs(x2) < 0.5f0 && abs(x3) < 0.5f0 ? one(x1) : zero(x1)
    return SVector(u)
end
initial_condition = initial_condition_heaviside_step

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Create a uniformly refined mesh with periodic boundaries
coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinate
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinate
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = true)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with given time span
ode = semidiscretize(semi, (0.0, 0.7))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
# We use a large CFL number here, which causes Zhang-Shu limiting by itself to fail.
stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

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
