
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5/3
equations = CompressibleEulerEquations2D(gamma)

initial_condition = initial_condition_blob

surface_flux = flux_lax_friedrichs
volume_flux  = flux_chandrashekar
basis = LobattoLegendreBasis(4)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.4,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=pressure)

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-20, -20)
coordinates_max = ( 20,  20)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=100_000,)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 8.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=1.0,
                                          alpha_min=0.0001,
                                          alpha_smooth=true,
                                          variable=pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      med_level =7, med_threshold=0.003,
                                      max_level =7, max_threshold=0.3)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.3)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

callbacks = CallbackSet(summary_callback, amr_callback, stepsize_callback, save_solution, analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
