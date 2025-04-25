using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################

elixir_file = "elixir_advection_unstructured_flag.jl"
trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))

###############################################################################
# Interpolate original solution to HIGHER order (3 -> 4)

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

restart_file = "restart_000000011.h5"
restart_filename = joinpath("out", restart_file)
tspan = (load_time(restart_filename), 0.4)
ode = semidiscretize(semi, tspan, restart_filename)

# We need to lower the CFL number compared to the k = 3 case
stepsize_callback = StepsizeCallback(cfl = 1.0)

analysis_callback = AnalysisCallback(semi, interval = 100)

callbacks = CallbackSet(summary_callback, # Re-used
                        analysis_callback, # Re-initialized
                        stepsize_callback) # Re-initialized

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
