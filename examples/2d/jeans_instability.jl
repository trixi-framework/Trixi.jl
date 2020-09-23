
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5/3
equations_euler = CompressibleEulerEquations2D(gamma)

# the initial conditions could also just be defined here
initial_conditions = Trixi.initial_conditions_jeans_instability

polydeg = 3
surface_flux = flux_hll
solver_euler = DGSEM(polydeg, surface_flux)

coordinates_min = (0, 0)
coordinates_max = (1, 1)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_conditions, solver_euler)


###############################################################################
# semidiscretization of the hyperbolic diffusion equations
resid_tol = 1.0e-4
equations_gravity = HyperbolicDiffusionEquations2D(resid_tol)

solver_gravity = DGSEM(polydeg)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_conditions, solver_gravity,
                                            source_terms=source_terms_harmonic)


###############################################################################
# combining both semidiscretizations for Euler + self-gravity
parameters = ParametersEulerGravity(background_density=1.5e7, # aka rho0
                                    gravitational_constant=6.674e-8, # aka G
                                    cfl=0.3,
                                    n_iterations_max=1000,
                                    timestep_gravity=timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)


###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

# TODO: Taal implement, printing stuff (Logo etc.) at the beginning (optionally)

# TODO: Taal debug
analysis_interval = 1
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi_euler, analysis_interval=analysis_interval,
                                     save_analysis=true,
                                     extra_analysis_integrals=(entropy, energy_total))
# TODO: Taal implement, energy_kinetic, energy_internal, energy_potential
# TODO: Taal implement, analysis specific to Euler+gravity ?

save_solution = SaveSolutionCallback(solution_interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

# TODO: Taal, IO
# restart_interval = 10

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(stepsize_callback, analysis_callback, save_solution, alive_callback)


###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);

nothing
