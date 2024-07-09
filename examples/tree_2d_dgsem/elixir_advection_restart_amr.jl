
using OrdinaryDiffEq
using Trixi

###############################################################################
# Define time integration algorithm
alg = CarpenterKennedy2N54(williamson_condition = false)
# Create a restart file
trixi_include(@__MODULE__, joinpath(@__DIR__, "elixir_advection_extended.jl"), alg = alg,
              tspan = (0.0, 3.0))

###############################################################################
# adapt the parameters that have changed compared to "elixir_advection_extended.jl"

# Note: If you get a restart file from somewhere else, you need to provide
# appropriate setups in the elixir loading a restart file

restart_filename = joinpath("out", "restart_000000040.h5")
mesh = load_mesh(restart_filename)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (load_time(restart_filename), 5.0)
dt = load_dt(restart_filename)
ode = semidiscretize(semi, tspan, restart_filename);

# Do not overwrite the initial snapshot written by elixir_advection_extended.jl.
save_solution.condition.save_initial_solution = false

# Add AMR callback
amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 3,
                                      med_level = 4, med_threshold = 0.8,
                                      max_level = 5, max_threshold = 1.2)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)
callbacks_ext = CallbackSet(amr_callback, callbacks.discrete_callbacks...)

integrator = init(ode, alg,
                  dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks_ext, maxiters = 100_000)

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
