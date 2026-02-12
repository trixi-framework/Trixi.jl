using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the invsicid Burgers' equation with diffusion

equations = InviscidBurgersEquation1D()

diffusivity() = 1e-3
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

coordinates_min = -0.1
coordinates_max = 0.2

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 30_000,
                periodicity = false)

# This solution comprises a compression wave followed by a rarefaction wave
# which is damped out over time.
function initial_condition_n_wave(x, t, equations)
    t0 = 1e-2 # start at t0 to avoid singularity at t=0
    damping = exp(-x[1]^2 / (4 * diffusivity() * (t + t0)))

    wave_speed = 1.0
    u = x[1] / (t + t0) * damping + wave_speed

    return SVector(u)
end
initial_condition = initial_condition_n_wave

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = boundary_condition
boundary_conditions_parabolic = boundary_condition

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 100)

alive_callback = AliveCallback(analysis_interval = 100)

# Timestep is limited by standard/advective/convective CFL
stepsize_callback = StepsizeCallback(cfl = 0.6, cfl_diffusive = 0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
