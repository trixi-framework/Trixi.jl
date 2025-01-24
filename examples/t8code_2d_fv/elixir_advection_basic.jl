using OrdinaryDiffEq
using Trixi

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

solver = FV(order = 2, extended_reconstruction_stencil = false,
            surface_flux = flux_lax_friedrichs)

# Option 1: coordinates
coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (8.0, 8.0) # maximum coordinates (max(x), max(y))

mapping_coordinates = Trixi.coordinates2mapping(coordinates_min, coordinates_max)

# Option 2: faces
# waving flag
# f1(s) = SVector(-1.0, s - 1.0)
# f2(s) = SVector(1.0, s + 1.0)
# f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
# f4(s) = SVector(s, 1.0 + sin(0.5 * pi * s))

# [0,8]^2
f1(s) = SVector(0.0, 4 * (s + 1))
f2(s) = SVector(8.0, 4 * (s + 1))
f3(s) = SVector(4 * (s + 1), 0.0)
f4(s) = SVector(4 * (s + 1), 8.0)
faces = (f1, f2, f3, f4)
Trixi.validate_faces(faces)
mapping_faces = Trixi.transfinite_mapping(faces)

# Option 3: classic mapping
function mapping(xi, eta)
    x = 4 * (xi + 1)
    y = 4 * (eta + 1)

    return SVector(x, y)
end

trees_per_dimension = (2, 2)

# Disabling the gc for almost the entire elixir seems to work in order to fix the SegFault errors with trixi_t8_mapping_c and mapping_coordinates
# It's also possible to move this to the constructor.
GC.enable(false)

# Notes:
# Life time issue for the GC tracked Julia object used in C. Only with coordinates_min/max
# - GC enabled:
#   o for mapping, mapping_faces: every three options work
#   o trixi_t8_mapping_c from elixir: `mapping_coordinates` doesn't work. SegFaults when evaluate geometry
#   o trixi_t8_mapping_c() from file: `mapping_coordinates` doesn't work. SegFaults when evaluate geometry
#   o pass mapping_coordinates doesn't work. SegFaults when evaluate geometry
#   o pass coordinates_min/max doesn't work. SegFaults when evaluate geometry
# - GC disabled as in elixir:
#   o Everything seems to work

element_class = :quad
mesh = T8codeMesh(trees_per_dimension, element_class,
                  # mapping = Trixi.trixi_t8_mapping_c(mapping_coordinates),
                  # Plan is to use either
                  coordinates_max = coordinates_max, coordinates_min = coordinates_min,
                  # or at least
                  # mapping = mapping_coordinates,
                  initial_refinement_level = 6)

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

GC.enable(true)
