
using OrdinaryDiffEq
using Trixi

###############################################################################
# create a restart file

trixi_include(@__MODULE__, joinpath(@__DIR__, "elixir_advection_extended.jl"))

###############################################################################
# adapt the parameters that have changed compared to "elixir_advection_extended.jl"

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

restart_filename = joinpath("out", "restart_000000019.h5")
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (load_time(restart_filename), 2.0)
dt = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename);

# Do not overwrite the initial snapshot written by elixir_advection_extended.jl.
save_solution.condition.save_initial_solution = false

integrator = init(ode, CarpenterKennedy2N54(williamson_condition = false),
                  dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks, maxiters = 100_000);

# Get the last time index and work with that.
load_timestep!(integrator, restart_filename)

###############################################################################
# run the simulation

sol = solve!(integrator)
summary_callback() # print the timer summary
