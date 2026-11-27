using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################

elixir_file = "elixir_euler_free_stream.jl"
trixi_include(@__MODULE__, joinpath(@__DIR__, elixir_file))

restart_file = "restart_000000748.h5"
restart_filename = joinpath("out", restart_file)
tspan = (load_time(restart_filename), 2.0)

###############################################################################
# Interpolate original solution to HIGHER order (3 -> 4), also exchange surface flux

solver = DGSEM(polydeg = 4, surface_flux = flux_hll)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = (;
                                                           all = BoundaryConditionDirichlet(initial_condition)))

# On curvilinear meshes, the conversion of the solution to a different polynomial degree
# does in general not conserve the discrete mass of each element exactly.
# Use `correct_mass = true` to restore exact conservation.
# This may come at the cost of losing free-stream preservation
# if the discrete volume of an element depends on the polynomial degree,
# see the docstring of `semidiscretize`.
ode = semidiscretize(semi, tspan, restart_filename,
                     correct_mass = false)

# We need to lower the CFL number compared to the k = 3 case
stepsize_callback = StepsizeCallback(cfl = 1.0)

# Reinitialize the `AnalysiCallback` for changed basis polynomial degree
analysis_callback = AnalysisCallback(semi, interval = 100)

callbacks = CallbackSet(summary_callback, # Re-used
                        analysis_callback, # Re-initialized
                        stepsize_callback) # Re-initialized

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, ode_default_options()..., callback = callbacks);
