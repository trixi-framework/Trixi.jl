using Trixi

path = "out"

restart_file = "restart_063585.h5"
restart_filename = joinpath(path, restart_file)

new_path = path

tspan = (load_time(restart_filename), 6.7)
ode = semidiscretize(semi, tspan, restart_filename);

save_solution = SaveSolutionCallback(output_directory = path,
                                     interval = 5000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

save_restart = SaveRestartCallback(output_directory = path,
                                   interval = 50000,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        save_restart, save_solution)

sol = Trixi.solve(ode,
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
