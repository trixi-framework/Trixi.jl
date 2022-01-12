
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleMoistEulerEquations2D()

initial_condition = initial_condition_warm_bubble

boundary_condition = (x_neg=boundary_condition_periodic,
                      x_pos=boundary_condition_periodic,
                      y_neg=boundary_condition_slip_wall,
                      y_pos=boundary_condition_slip_wall)

source_term = source_terms_warm_bubble

###############################################################################
# Get the DG approximation space


polydeg = 4
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_LMARS
volume_flux = flux_shima_etal

indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

#volume_integral=VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-5000.0, 0.0)
coordinates_max = (5000.0, 10000.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                periodicity=(true, false),
                n_cells_max=40_000)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition,
                                    source_terms=source_term)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 500.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
solution_variables = cons2drypot

analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=solution_variables)


#amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=velocity),
#                                      base_level=3, max_level=6,
#                                      med_threshold=1, max_threshold=5)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=velocity),
                                      base_level=3, max_level=5,
                                      med_threshold=0.2, max_threshold=1.0)

amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=false)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
#                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

#limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
#                                               variables=(Trixi.density, pressure))
#stage_limiter! = limiter!
#step_limiter!  = limiter!

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
