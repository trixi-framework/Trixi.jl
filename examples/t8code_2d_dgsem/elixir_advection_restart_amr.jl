using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# create a restart file

elixir_file = "elixir_advection_extended.jl"
restart_file = "restart_000000021.h5"

trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))

###############################################################################
# adapt the parameters that have changed compared to "elixir_advection_extended.jl"

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

restart_filename = joinpath("out", restart_file)
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (load_time(restart_filename), 2.0)
dt = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename)

# Do not overwrite the initial snapshot written by elixir_advection_extended.jl.
save_solution.condition.save_initial_solution = false

# Add AMR callback
amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 0,
                                      med_level = 0, med_threshold = 0.8,
                                      max_level = 1, max_threshold = 1.2)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true,
                           dynamic_load_balancing = false)
# We disable `dynamic_load_balancing` for now, since t8code does not support
# partitioning for coarsening yet. That is, a complete family of elements always
# stays on rank and is not split up due to partitioning. Without this feature
# dynamic AMR simulations are not perfectly deterministic regarding to
# convergent tests. Once this feature is available in t8code load balancing is
# enabled again.

callbacks_ext = CallbackSet(amr_callback, callbacks.discrete_callbacks...)

integrator = init(ode, CarpenterKennedy2N54(williamson_condition = false);
                  dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()..., callback = callbacks_ext, maxiters = 100_000);

# Get the last time index and work with that.
load_timestep!(integrator, restart_filename)

###############################################################################
# run the simulation

sol = solve!(integrator)
summary_callback() # print the timer summary
