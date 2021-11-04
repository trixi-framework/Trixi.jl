
using OrdinaryDiffEq
using Trixi

###############################################################################
# create a restart file

elixir_file = "elixir_advection_extended.jl"
restart_file = "restart_000021.h5"

trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))


###############################################################################
# adapt the parameters that have changed compared to "elixir_advection_extended.jl"

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

restart_filename = joinpath("out", restart_file)
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (load_time(restart_filename), 2.0)
ode = semidiscretize(semi, tspan, restart_filename);


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
