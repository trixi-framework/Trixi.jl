using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Define time integration algorithm
alg = CarpenterKennedy2N54(williamson_condition = false)
# Create a restart file
base_elixir = "elixir_advection_extended.jl"
trixi_include(@__MODULE__, joinpath(@__DIR__, base_elixir), alg = alg)

restart_filename = joinpath("out", "restart_000000018.h5")
tspan = (1.0, 1.0)

###############################################################################
# Convert original solution to LOWER order (3 -> 2)

# Set `interpolate_high2low = false` to construct the solution via L2 projection
# instead of interpolation. The L2 projection conserves the integral over each element
# and yields smaller L2 errors, but is in general not continuous across element interfaces.
interpolate_high2low = true

solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

ode = semidiscretize(semi, tspan, restart_filename; interpolate_high2low)

# We can increase the CFL number compared to the k = 3 case
stepsize_callback = StepsizeCallback(cfl = 2.0)

# Reinitialize the `AnalysisCallback` for changed basis polynomial degree
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

callbacks = CallbackSet(summary_callback, # Re-used
                        alive_callback, # Re-used
                        analysis_callback, # Re-initialized
                        stepsize_callback) # Re-initialized

sol = solve(ode, alg;
            dt = 1.0, ode_default_options()..., callback = callbacks);
