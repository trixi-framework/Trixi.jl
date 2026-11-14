using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# create a restart file

elixir_file = "elixir_advection_extended.jl"
restart_file = "restart_000000020.h5"

# Disable the in-situ visualization of the base elixir, which would open a plot window
trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file),
              visualization = TrivialCallback())

restart_filename = joinpath("out", restart_file)
tspan = (load_time(restart_filename), 2.0)

###############################################################################
# Project original solution to LOWER order (3 -> 2)

solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

# By default, the higher-degree polynomial is L2-projected onto the lower-degree polynomial,
# which conserves the integral over each element.
# Set `interpolate_high2low = true` to interpolate instead,
# thereby preserving continuity across cell interfaces.
interpolate_high2low = false
ode = semidiscretize(semi, tspan, restart_filename; interpolate_high2low)

# We can increase the CFL number compared to the k = 3 case
stepsize_callback = StepsizeCallback(cfl = 2.0)

# Reinitialize the `AnalysisCallback` for changed basis polynomial degree
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy, energy_total))

callbacks = CallbackSet(summary_callback, # Re-used
                        alive_callback, # Re-used
                        analysis_callback, # Re-initialized
                        stepsize_callback) # Re-initialized

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
