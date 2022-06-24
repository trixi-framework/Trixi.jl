
using OrdinaryDiffEq
using Trixi

###############################################################################
# create a restart file

trixi_include(@__MODULE__, joinpath(@__DIR__, "elixir_euler_astro_jet_MCL.jl"))

###############################################################################
# adapt the parameters that have changed compared to "elixir_euler_astro_jet_MCL.jl"

restart_filename = joinpath("out", "restart_000001.h5")
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)

tspan = (load_time(restart_filename), 0.001)
ode = semidiscretize(semi, tspan, restart_filename);

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=false,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)
###############################################################################
# run the simulation
sol = Trixi.solve(ode,
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback=callbacks);
summary_callback() # print the timer summary
