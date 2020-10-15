
using OrdinaryDiffEq
using Trixi

##
# OBS! This crashes due to an issue with the combination of shockcapturing + non-conservative terms
#      It can be run if "have_nonconservative_terms(::IdealGlmMhdEquations2D) = Val(true)" is set
#      to false in the ideal_glm_mhd.jl

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations2D(1.4)

initial_conditions = initial_conditions_blast_wave

surface_flux = flux_lax_friedrichs
volume_flux  = flux_central
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, -0.5)
coordinates_max = ( 0.5,  0.5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_conditions, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.01)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

indicator_amr = IndicatorHennemannGassner(semi,
                                          alpha_max=0.5,
                                          alpha_min=0.001,
                                          alpha_smooth=true,
                                          variable=density_pressure)
amr_indicator = IndicatorThreeLevel(semi, indicator_amr,
                                    base_level=4,
                                    max_level =6, max_threshold=0.01)
amr_callback = AMRCallback(semi, amr_indicator,
                           interval=5,
                           adapt_initial_conditions=true,
                           adapt_initial_conditions_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.5) # can probably be increased when shock-capturing is fixed for MHD

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
