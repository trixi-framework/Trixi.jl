
using OrdinaryDiffEq
using Trixi

###############################################################################
# Define time integration algorithm
alg = CarpenterKennedy2N54(williamson_condition = false)
# Create a restart file
trixi_include(@__MODULE__, joinpath(@__DIR__, "elixir_advection_extended.jl"), alg = alg,
              tspan = (0.0, 10.0))

###############################################################################
# adapt the parameters that have changed compared to "elixir_advection_extended.jl"

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

restart_filename = joinpath("out", "restart_000000040.h5")
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (load_time(restart_filename), 10.0)
dt = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename);

# Do not overwrite the initial snapshot written by elixir_advection_extended.jl.
save_solution.condition.save_initial_solution = false

integrator = init(ode, alg,
                  dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks;
                  ode_default_options()...); # default options because an adaptive time stepping method is used in test_mpi_tree.jl

# Load saved context for adaptive time integrator
if integrator.opts.adaptive
    load_adaptive_time_integrator!(integrator, restart_filename)
end

# Get the last time index and work with that.
load_timestep!(integrator, restart_filename)

###############################################################################
# run the simulation

sol = solve!(integrator)

summary_callback() # print the timer summary
