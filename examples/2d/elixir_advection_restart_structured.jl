
using OrdinaryDiffEq
using Trixi

###############################################################################
# create a restart file

trixi_include(@__MODULE__, joinpath(@__DIR__, "elixir_advection_extended_structured.jl"))


###############################################################################
# adapt the parameters that have changed compared to "elixir_advection_extended.jl"

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

restart_filename = joinpath("out", "restart_000071.h5")
mesh = load_mesh(restart_filename, n_cells_max=0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (load_time(restart_filename), 2.0)
ode = semidiscretize(semi, tspan, restart_filename);


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
