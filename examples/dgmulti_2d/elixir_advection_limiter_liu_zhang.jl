using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (1.0, 0.5)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Step function initial condition which is 1 on [-2/3, 2/3]^2 and zero elsewhere.
# The discontinuity does not coincide with element interfaces, which lie at multiples of
# 1/4. Otherwise, nodes would lie on the discontinuity, and whether they are set to 0 or 1
# would depend on roundoff in the node coordinates and thus on the platform.
function initial_condition_heaviside_step(x, t,
                                          equations::LinearScalarAdvectionEquation2D)
    x1, x2 = x
    u = abs(x1) < 2 / 3 && abs(x2) < 2 / 3 ? one(x1) : zero(x1)
    return SVector(u)
end
initial_condition = initial_condition_heaviside_step

# DG solver on triangles with SBP nodes and (local) Lax-Friedrichs/Rusanov surface flux
polydeg = 3
dg = DGMulti(polydeg = polydeg, element_type = Tri(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

cells_per_dimension = (8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension;
                   coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),
                   periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.7)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

# We use a large CFL number here, which causes Zhang-Shu limiting by itself to fail.
cfl = 1.6
stepsize_callback = StepsizeCallback(cfl = cfl)

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
                          step_limiter! = global_limiter!,
                          thread = Trixi.Threaded());
            adaptive = false,
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
