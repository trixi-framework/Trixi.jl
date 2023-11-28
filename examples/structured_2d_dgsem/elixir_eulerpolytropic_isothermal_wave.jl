
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the polytropic Euler equations

gamma = 1.0     # With gamma = 1 the system is isothermal.
kappa = 1.0     # Scaling factor for the pressure.
equations = PolytropicEulerEquations2D(gamma, kappa)

# Linear pressure wave in the negative x-direction.
function initial_condition_wave(x, t, equations::PolytropicEulerEquations2D)
    rho = 1.0
    v1 = 0.0
    if x[1] > 0.0
        rho = ((1.0 + 0.01 * sin(x[1] * 2 * pi)) / equations.kappa)^(1 / equations.gamma)
        v1 = ((0.01 * sin((x[1] - 1 / 2) * 2 * pi)) / equations.kappa)
    end
    v2 = 0.0

    return prim2cons(SVector(rho, v1, v2), equations)
end
initial_condition = initial_condition_wave

volume_flux = flux_winters_etal
solver = DGSEM(polydeg = 2, surface_flux = flux_hll,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-2.0, -1.0)
coordinates_max = (2.0, 1.0)

cells_per_dimension = (64, 32)

boundary_conditions = (x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

mesh = StructuredMesh(cells_per_dimension,
                      coordinates_min,
                      coordinates_max)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 50,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-4, 1.0e-4),
                                                     variables = (Trixi.density, pressure))

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
