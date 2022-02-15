
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_weak_blast_wave

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
volume_flux = flux_ranocha
indicator_sc = IndicatorIDP(equations, basis; variable=Trixi.density)
volume_integral=VolumeIntegralStaggeredGrid(indicator_sc; volume_flux_dg=volume_flux,
                                                          volume_flux_fv=volume_flux)
solver = DGSEM(basis, flux_ranocha, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=10_000,
                periodicity=true)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_condition_periodic)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback, save_solution,
                        analysis_callback, alive_callback,
                        stepsize_callback)


###############################################################################
# run the simulation
# Trixi.solve(ode, Trixi.CarpenterKennedy2N43();
#                dt=0.01, callback=callbacks)
# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = Trixi.solve_IDP(ode, semi;
            dt=0.01, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary
