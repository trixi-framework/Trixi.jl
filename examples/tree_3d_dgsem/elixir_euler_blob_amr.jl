
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5/3
equations = CompressibleEulerEquations3D(gamma)

initial_condition = initial_condition_blob

volume_flux = flux_ranocha
solver = DGSEM(polydeg=3, surface_flux=flux_hllc,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-20.0, -20.0, -20.0)
coordinates_max = ( 20.0,  20.0,  20.0)

refinement_patches = (
  (type="box", coordinates_min=(-20.0, -10.0, -10.0), coordinates_max=(-10.0, 10.0, 10.0)),
  (type="box", coordinates_min=(-20.0,  -5.0,  -5.0), coordinates_max=(-10.0,  5.0,  5.0)),
  (type="box", coordinates_min=(-17.0,  -2.0,  -2.0), coordinates_max=(-13.0,  2.0,  2.0)),
  (type="box", coordinates_min=(-17.0,  -2.0,  -2.0), coordinates_max=(-13.0,  2.0,  2.0)),
)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                refinement_patches=refinement_patches,
                n_cells_max=100_000,)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=200,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_indicator = IndicatorLöhner(semi,
                                variable=Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=1,
                                      med_level =0, med_threshold=0.1, # med_level = current level
                                      max_level =6, max_threshold=0.3)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=3,
                           adapt_initial_condition=false,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=1.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        amr_callback, stepsize_callback)


limiter! = PositivityPreservingLimiterZhangShu(thresholds=(1.0e-4, 1.0e-4),
                                               variables=(Trixi.density, pressure))
stage_limiter! = limiter!
step_limiter!  = limiter!

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, step_limiter!, williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
