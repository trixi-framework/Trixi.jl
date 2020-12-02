
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

equations = LatticeBoltzmannEquation2D(Ma=0.1, Re=1000)

initial_condition = initial_condition_lid_driven_cavity
boundary_conditions = (
                       x_neg=boundary_condition_wall_noslip,
                       x_pos=boundary_condition_wall_noslip,
                       y_neg=boundary_condition_wall_noslip,
                       y_pos=boundary_condition_lid_driven_cavity,
                      )

surface_flux = flux_lax_friedrichs
solver = DGSEM(5, surface_flux)

coordinates_min = (0, 0)
coordinates_max = (1, 1)
# refinement_patches = (
#   (type="box", coordinates_min=(0.0, -1.0), coordinates_max=(1.0, 1.0)),
# )
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                # refinement_patches=refinement_patches,
                periodicity=false,
                n_cells_max=10_000,)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=1000,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=1000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

stepsize_callback = StepsizeCallback(cfl=1.0)

collision_callback = LBMCollisionCallback()

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback, 
                        save_restart, save_solution,
                        stepsize_callback,
                        collision_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
