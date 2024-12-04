using OrdinaryDiffEq
using Trixi

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

initial_condition = initial_condition_convergence_test

solver = FV(order = 2, extended_reconstruction_stencil = false,
            surface_flux = flux_lax_friedrichs)

# Option 1: coordinates
coordinates_min = (0.0, 0.0, 0.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (8.0, 8.0, 8.0) # maximum coordinates (max(x), max(y), max(z))

mapping_coordinates = Trixi.coordinates2mapping(coordinates_min, coordinates_max)

# Option 3: classic mapping
function mapping(xi, eta, zeta)
    x = 4 * (xi + 1)
    y = 4 * (eta + 1)
    z = 4 * (zeta + 1)

    return SVector(x, y, z)
end

trees_per_dimension = (2, 2, 2)

# For explanations, see 2D elixir.
GC.enable(false)

element_class = :hex
mesh = T8codeMesh(trees_per_dimension, element_class;
                  # mapping = Trixi.trixi_t8_mapping_c(mapping),
                  # Plan is to use either
                  coordinates_max = coordinates_max, coordinates_min = coordinates_min,
                  # or at least
                  # mapping = mapping,
                  initial_refinement_level = 5)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback()

# Note: Since the mesh must be finalizized by hand in the elixir, it is not defined anymore here.
# Moved allocation test to the elixirs for now.
using Test
let
    t = sol.t[end]
    u_ode = sol.u[end]
    du_ode = similar(u_ode)
    @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
end

GC.enable(true)

# Finalize `T8codeMesh` to make sure MPI related objects in t8code are
# released before `MPI` finalizes.
!isinteractive() && finalize(mesh)
