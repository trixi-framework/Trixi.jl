using OrdinaryDiffEq
using Trixi

trixi_include("idp_density_entropy_cfl0.9_t3.7.jl")

restart_filename = "out_t3_7/restart_062276.h5"

tspan = (load_time(restart_filename), 6.7)
ode = semidiscretize(semi, tspan, restart_filename);

save_solution = SaveSolutionCallback(output_directory="out_t6_7/",
                                     interval=5000,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

save_restart = SaveRestartCallback(output_directory="out_t6_7/",
                                   interval=50000,
                                   save_final_restart=true)

callbacks = CallbackSet(summary_callback,
                        stepsize_callback,
                        analysis_callback, alive_callback,
                        save_restart,
                        save_solution)

sol = Trixi.solve(ode,
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback=callbacks);
summary_callback() # print the timer summary