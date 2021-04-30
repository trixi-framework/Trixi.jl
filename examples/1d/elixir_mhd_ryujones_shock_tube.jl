
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
gamma = 5/3
equations = IdealGlmMhdEquations1D(gamma)

initial_condition = initial_condition_ryujones_shock_tube

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_hll
volume_flux  = flux_derigs_etal
basis = LobattoLegendreBasis(3)

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=Trixi.density)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = 0
coordinates_max = 1
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,
                n_cells_max=10_000,
                periodicity=false)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# amr_indicator = IndicatorHennemannGassner(semi,
#                                           alpha_max=0.5,
#                                           alpha_min=0.001,
#                                           alpha_smooth=true,
#                                           variable=Trixi.density)
# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level=5,
#                                       max_level=8, max_threshold=0.01)
# amr_callback = AMRCallback(semi, amr_controller,
#                            interval=5,
#                            adapt_initial_condition=true,
#                            adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
