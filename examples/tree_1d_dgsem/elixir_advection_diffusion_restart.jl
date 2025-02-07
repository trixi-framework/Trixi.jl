using OrdinaryDiffEqSDIRK, ADTypes
using Trixi

###############################################################################
# create a restart file

elixir_file = "elixir_advection_diffusion.jl"
trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))

###############################################################################
# initialize the ODE

restart_file = "restart_000000018.h5"
restart_filename = joinpath("out", restart_file)
tspan = (load_time(restart_filename), 2.0)

ode = semidiscretize(semi, tspan, restart_filename)

# Do not save restart files here
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, KenCarp4(autodiff = AutoFiniteDiff()), abstol = time_abs_tol, reltol = time_int_tol,
            save_everystep = false, callback = callbacks)

# Print the timer summary
summary_callback()
