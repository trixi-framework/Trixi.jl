using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# create a restart file

elixir_file = "elixir_advection_extended.jl"
restart_file = "restart_000000020.h5"

trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))

restart_filename = joinpath("out", restart_file)
tspan = (load_time(restart_filename), 2.0)

###############################################################################
# Interpolate original solution to LOWER order (3 -> 2)

solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

# By default, the higher-degree polynomial is interpolated to the lower degree polynomial,
# thereby preserving continuity across cell interfaces.
# Set `interpolate_high2low = false` to avoid interpolation in favor of L2-projection.
ode = semidiscretize(semi, tspan, restart_filename)

# We can increase the CFL number compared to the k = 3 case
stepsize_callback = StepsizeCallback(cfl = 2.0)

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy, energy_total))

callbacks = CallbackSet(summary_callback, # Re-used
                        alive_callback, # Re-used
                        analysis_callback, # Re-initialized
                        stepsize_callback) # Re-initialized

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
