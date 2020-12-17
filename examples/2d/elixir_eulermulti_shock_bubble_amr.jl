using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler multicomponent equations
gamma1              = 1.4
gamma2              = 1.648
gas_constant_1      = 0.287
gas_constant_2      = 1.578
cv1                 = 0.72
cv2                 = 2.44
cp1                 = 1.007 # rs1 + cvs1, Check: gamma1 = cps1/cvs1 = 1.4
cp2                 = 4.018 # rs2 + cvs2, Check: gamma2 = cps2/cvs2 = 1.648
equations           = CompressibleEulerMulticomponentEquations2D(gamma1, gamma2, gas_constant_1, gas_constant_2, cv1, cv2, cp1, cp2)

initial_condition   = initial_condition_shock_bubble

surface_flux        = flux_lax_friedrichs
volume_flux         = flux_chandrashekar
basis               = LobattoLegendreBasis(3)
indicator_sc        = IndicatorHennemannGassner(equations, basis,
                                                alpha_max=0.5,
                                                alpha_min=0.001,
                                                alpha_smooth=true,
                                                variable=density_pressure)
volume_integral     = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg=volume_flux,
                                                     volume_flux_fv=surface_flux)
solver              = DGSEM(basis, surface_flux, volume_integral)

coordinates_min     = (-2.25, -2.225)
coordinates_max     = ( 2.20,  2.225)
mesh                = TreeMesh(coordinates_min, coordinates_max,
                               initial_refinement_level=5,
                               n_cells_max=1_000_000)

semi                = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan               = (0.0, 0.012) # 0.0021... or until 0.015 for whole calc.
ode                 = semidiscretize(semi, tspan)

summary_callback    = SummaryCallback()

analysis_interval   = 1600 # change to 200+ for whole calc.
analysis_callback   = AnalysisCallback(semi, interval=analysis_interval)

alive_callback      = AliveCallback(analysis_interval=analysis_interval)

save_solution       = SaveSolutionCallback(interval=1600,     # 40 or change to 200+ for whole calc.  
                                           save_initial_solution=true,
                                           save_final_solution=true,
                                           solution_variables=:primitive)

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=1.0,
                                          alpha_min=0.0001,
                                          alpha_smooth=true,
                                          variable=density_pressure)

amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                          base_level=5,
                                          max_level =12, max_threshold=0.01)

amr_callback = AMRCallback(semi, amr_controller,
                               interval=5,
                               adapt_initial_condition=true,
                               adapt_initial_condition_only_refine=true)

stepsize_callback   = StepsizeCallback(cfl=0.3)

callbacks           = CallbackSet(summary_callback,
                                  analysis_callback, 
                                  alive_callback, 
                                  save_solution,
                                  amr_callback,
                                  stepsize_callback)


###############################################################################
# run the simulation
sol                 = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                                                      dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                                                      save_everystep=false, 
                                                      callback=callbacks);
summary_callback() # print the timer summary