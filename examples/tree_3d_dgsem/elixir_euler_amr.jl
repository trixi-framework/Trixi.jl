
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

"""
    initial_condition_density_pulse(x, t, equations::CompressibleEulerEquations3D)

A Gaussian pulse in the density with constant velocity and pressure; reduces the
compressible Euler equations to the linear advection equations.
"""
function initial_condition_density_pulse(x, t, equations::CompressibleEulerEquations3D)
    rho = 1 + exp(-(x[1]^2 + x[2]^2 + x[3]^2)) / 2
    v1 = 1
    v2 = 1
    v3 = 1
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    rho_v3 = rho * v3
    p = 1
    rho_e = p / (equations.gamma - 1) + 1 / 2 * rho * (v1^2 + v2^2 + v3^2)
    return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e)
end
initial_condition = initial_condition_density_pulse
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-5.0, -5.0, -5.0)
coordinates_max = (5.0, 5.0, 5.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 4,
                                      med_level = 5, med_threshold = 1.05,
                                      max_level = 6, max_threshold = 1.3)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        amr_callback, stepsize_callback);

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
