using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler multicomponent equations
gamma1              = 1.4
gamma2              = 1.648
gas_constant1       = 0.287
gas_constant2       = 1.578
equations           = CompressibleEulerMulticomponentEquations2D(gamma1, gamma2, gas_constant1, gas_constant2)

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
                               initial_refinement_level=7,
                               n_cells_max=1_000_000)

semi                = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan               = (0.0, 0.012) # 0.0021..
ode                 = semidiscretize(semi, tspan)

summary_callback    = SummaryCallback()

analysis_interval   = 300 # change to 200+ for whole calc.
analysis_callback   = AnalysisCallback(semi, interval=analysis_interval)

alive_callback      = AliveCallback(analysis_interval=analysis_interval)

save_solution       = SaveSolutionCallback(interval=300,     # 40 or change to 200+ for whole calc.  
                                           save_initial_solution=true,
                                           save_final_solution=true,
                                           solution_variables=:primitive)

stepsize_callback   = StepsizeCallback(cfl=0.3)

callbacks           = CallbackSet(summary_callback,
                                  analysis_callback, 
                                  alive_callback, 
                                  save_solution,
                                  stepsize_callback)


###############################################################################
# run the simulation
sol                 = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                                                      dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                                                      save_everystep=false, 
                                                      callback=callbacks, 
                                                      maxiters=1e5);
summary_callback() # print the timer summary