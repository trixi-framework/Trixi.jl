
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

initial_condition = initial_condition_khi

surface_flux = flux_lax_friedrichs
#surface_flux = flux_hllc
#volume_flux  = flux_chandrashekar
#volume_flux  = flux_shima_etal
#volume_flux  = flux_kennedy_gruber
volume_flux  = flux_ranocha
#volume_flux  = flux_central
polydeg = 4
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.002,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
#volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                 volume_flux_dg=volume_flux,
#                                                 volume_flux_fv=surface_flux)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=200,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback, 
                        save_solution)#,
                        #stepsize_callback)

 #limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
 #                                               variables=(Trixi.density, pressure))
 #stage_limiter! = limiter!
 #step_limiter!  = limiter!


###############################################################################
# run the simulation

#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#            save_everystep=false, callback=callbacks);
#sol = solve(ode, SSPRK43(stage_limiter!,step_limiter!),
sol = solve(ode, SSPRK43(),
            #dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
#sol = solve(ode, RDPK3SpFSAL49(), abstol=1.0e-8, reltol=1.0e-8,
#            dt = 4.0e-3, save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
