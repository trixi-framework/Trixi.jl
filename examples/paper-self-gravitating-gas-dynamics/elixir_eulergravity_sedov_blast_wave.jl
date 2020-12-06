
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations_euler = CompressibleEulerEquations2D(gamma)

initial_condition = initial_condition_sedov_self_gravity
boundary_conditions = boundary_condition_sedov_self_gravity

surface_flux = flux_hll
volume_flux  = flux_chandrashekar
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations_euler, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver_euler = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-4, -4)
coordinates_max = ( 4,  4)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=100_000,
                periodicity=false)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition, solver_euler,
                                          boundary_conditions=boundary_conditions)


###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations2D()

# TODO: Taal, define initial/boundary conditions here for gravity?

solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition, solver_gravity,
                                            boundary_conditions=boundary_conditions,
                                            source_terms=source_terms_harmonic)


###############################################################################
# combining both semidiscretizations for Euler + self-gravity
parameters = ParametersEulerGravity(background_density=0.0, # aka rho0
                                    gravitational_constant=6.674e-8, # aka G
                                    cfl=2.4,
                                    resid_tol=1.0e-4,
                                    n_iterations_max=100,
                                    timestep_gravity=timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)


###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max=1.0,
                                          alpha_min=0.0,
                                          alpha_smooth=false,
                                          variable=density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=2,
                                      max_level =8, max_threshold=0.0003)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=1,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=1.0)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)

analysis_callback = AnalysisCallback(semi_euler, interval=analysis_interval,
                                     save_analysis=true,
                                     extra_analysis_integrals=(energy_total, energy_kinetic, energy_internal))

# TODO: Taal decide, first AMR or save solution etc.
callbacks = CallbackSet(summary_callback, amr_callback, stepsize_callback,
                        save_restart, save_solution,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
