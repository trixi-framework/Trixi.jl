using OrdinaryDiffEq
using Trixi

###############################################################################
# create timestep controller
controller = PIController(7 // 30, 2 // 15)

# create a restart file
trixi_include(@__MODULE__, joinpath(@__DIR__, "elixir_euler_density_wave_extended.jl"),
              tspan = (0.0, 2.0), controller = controller)

###############################################################################
# adapt the parameters that have changed compared to "elixir_euler_density_wave_extended.jl"

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

restart_filename = joinpath("out", "restart_001000.h5")
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (load_time(restart_filename), 4.0)
dt = load_dt(restart_filename)

ode = semidiscretize(semi, tspan, restart_filename);

# Do not overwrite the initial snapshot written by elixir_euler_density_wave_extended.jl.
save_solution.condition.save_initial_solution = false

alg = SSPRK43()
integrator = init(ode, alg,
                  dt = dt; 
                  save_everystep = false, callback = callbacks, controller = controller,
                  ode_default_options()...)
load_controller!(integrator, restart_filename)

# Get the last time index and work with that.
integrator.iter = load_timestep(restart_filename)
integrator.stats.naccept = integrator.iter

###############################################################################
# run the simulation

sol = solve!(integrator)

summary_callback() # print the timer summary
