
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the Lattice-Boltzmann equations for the D2Q9 scheme

equations = LatticeBoltzmannEquations2D(Ma=0.1, Re=1000)

initial_condition = initial_condition_lid_driven_cavity
boundary_conditions = (
                       x_neg=boundary_condition_noslip_wall,
                       x_pos=boundary_condition_noslip_wall,
                       y_neg=boundary_condition_noslip_wall,
                       y_pos=boundary_condition_lid_driven_cavity,
                      )

solver = DGSEM(polydeg=5, surface_flux=flux_godunov)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                periodicity=false,
                n_cells_max=10_000,)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 400.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2macroscopic)

stepsize_callback = StepsizeCallback(cfl=1.0)

collision_callback = LBMCollisionCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        collision_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
