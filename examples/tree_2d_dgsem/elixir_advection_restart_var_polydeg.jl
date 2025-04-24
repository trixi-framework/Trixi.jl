using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Define time integration algorithm
alg = CarpenterKennedy2N54(williamson_condition = false)
# Create a restart file
base_elixir = "elixir_advection_extended.jl"
trixi_include(@__MODULE__, joinpath(@__DIR__, base_elixir), alg = alg,
              tspan = (0.0, 10.0))

restart_filename = joinpath("out", "restart_000000040.h5")
tspan = (10.0, 11.0)

###############################################################################
# Interpolate original solution to HIGHER order (3 -> 4)

solver = DGSEM(polydeg = 4, surface_flux = flux_lax_friedrichs)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

ode = semidiscretize(semi, tspan, restart_filename)

# We need to lower the CFL number compared to the k = 3 case
stepsize_callback = StepsizeCallback(cfl = 1.0)

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy, energy_total))

callbacks = CallbackSet(summary_callback, # Re-used
                        alive_callback, # Re-used
                        analysis_callback, # Re-initialized
                        stepsize_callback) # Re-initialized

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);

###############################################################################
# Project original solution to LOWER order (3 -> 2)

solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

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
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
