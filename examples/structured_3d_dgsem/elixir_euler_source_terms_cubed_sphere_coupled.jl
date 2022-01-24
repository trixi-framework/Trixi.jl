
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test
boundary_condition = BoundaryConditionDirichlet(initial_condition)

solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)


# Note that half the elements will have left-handed coordinates.
# This allows for easier coupling. A version with only right-handed coordinates
# is used in meshes/p4est_mesh.jl.
function cubed_sphere_mapping(xi, eta, zeta, inner_radius, thickness, direction)
  alpha = xi * pi/4
  beta = eta * pi/4

  # Equiangular projection
  x = tan(alpha)
  y = tan(beta)

  # Coordinates on unit cube per direction
  cube_coordinates = [SVector(-1, x, y),
                      SVector( 1, x, y),
                      SVector(x, -1, y),
                      SVector(x,  1, y),
                      SVector(x, y, -1),
                      SVector(x, y,  1)]

  # Radius on cube surface
  r = sqrt(1 + x^2 + y^2)

  # Radius of the sphere
  R = inner_radius + thickness * (0.5 * (zeta + 1))

  # Projection onto the sphere
  R / r * cube_coordinates[direction]
end

function cubed_sphere_mapping(inner_radius, thickness, direction)
  mapping(xi, eta, zeta) = cubed_sphere_mapping(xi, eta, zeta, inner_radius, thickness, direction)
end

mapping_as_string_(inner_radius, thickness, direction) = """
function cubed_sphere_mapping(xi, eta, zeta, inner_radius, thickness, direction)
  alpha = xi * pi/4
  beta = eta * pi/4

  x = tan(alpha)
  y = tan(beta)

  r = sqrt(1 + x^2 + y^2)

  R = inner_radius + thickness * (0.5 * (zeta + 1))

  # Cube coordinates per direction
  cube_coordinates = [SVector(-1, x, y),
                      SVector( 1, x, y),
                      SVector(x, -1, y),
                      SVector(x,  1, y),
                      SVector(x, y, -1),
                      SVector(x, y,  1)]

  R / r * cube_coordinates[direction]
end; function cubed_sphere_mapping(inner_radius, thickness, direction)
  mapping(xi, eta, zeta) = cubed_sphere_mapping(xi, eta, zeta, inner_radius, thickness, direction)
end; mapping = cubed_sphere_mapping($inner_radius, $thickness, $direction)
"""

cells_per_dimension = (3, 3, 2)

inner_radius = 1
thickness = 0.1

  # Illustration of the local coordinates of each face. ξ and η are the first
  # local coordinates of each face. The third local coordinate ζ is always
  # pointing outwards, which allows for easy coupling.
  #               ┌────────────────────────────────────────────────────┐
  #              ╱│                                                   ╱│
  #             ╱ │                  ┌───> ξ                         ╱ │
  #            ╱  │                 ╱                               ╱  │
  #           ╱   │                V       4 (+y)                  ╱   │
  #          ╱    │               η                               ╱    │
  #         ╱     │                                              ╱     │
  #        ╱      │                                             ╱      │
  #       ╱       │                                            ╱       │
  #      ╱        │                                           ╱        │
  #     ╱         │                    η    5 (-z)           ╱         │
  #    ╱          │                    ↑                    ╱          │
  #   ╱           │                    │                   ╱           │
  #  ╱            │                    └───> ξ            ╱            │
  # ┌────────────────────────────────────────────────────┐    2 (+x)   │
  # │             │                                      │             │
  # │             │                                      │      ξ      │
  # │             │                                      │      ↑      │
  # │    1 (-x)   │                                      │      │      │
  # │             │                                      │      │      │
  # │      ξ      │                                      │     ╱       │
  # │      ↑      │                                      │    V        │
  # │      │      │                                      │   η         │
  # │      │      └──────────────────────────────────────│─────────────┘
  # │     ╱      ╱         η   6 (+z)                    │            ╱
  # │    V      ╱          ↑                             │           ╱
  # │   η      ╱           │                             │          ╱
  # │         ╱            └───> ξ                       │         ╱
  # │        ╱                                           │        ╱
  # │       ╱                                            │       ╱ Global coordinates:
  # │      ╱                                             │      ╱        y
  # │     ╱                      ┌───> ξ                 │     ╱         ↑
  # │    ╱                      ╱                        │    ╱          │
  # │   ╱                      V      3 (-y)             │   ╱           │
  # │  ╱                      η                          │  ╱            └─────> x
  # │ ╱                                                  │ ╱            ╱
  # │╱                                                   │╱            V
  # └────────────────────────────────────────────────────┘            z

mesh1 = StructuredMesh(cells_per_dimension, cubed_sphere_mapping(inner_radius, thickness, 1), periodicity=false,
                       mapping_as_string=mapping_as_string_(inner_radius, thickness, 1))

semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition, solver,
                                     source_terms=source_terms_convergence_test,
                                     boundary_conditions=(
                                       x_neg=BoundaryConditionCoupled(3, (:begin, :i_forward, :j_forward), Float64),
                                       x_pos=BoundaryConditionCoupled(4, (:begin, :i_forward, :j_forward), Float64),
                                       y_neg=BoundaryConditionCoupled(5, (:begin, :i_forward, :j_forward), Float64),
                                       y_pos=BoundaryConditionCoupled(6, (:begin, :i_forward, :j_forward), Float64),
                                       z_neg=boundary_condition,
                                       z_pos=boundary_condition,
                                     ))

mesh2 = StructuredMesh(cells_per_dimension, cubed_sphere_mapping(inner_radius, thickness, 2), periodicity=false,
                   mapping_as_string=mapping_as_string_(inner_radius, thickness, 2))

semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition, solver,
                                     source_terms=source_terms_convergence_test,
                                     boundary_conditions=(
                                       x_neg=BoundaryConditionCoupled(3, (:end, :i_forward, :j_forward), Float64),
                                       x_pos=BoundaryConditionCoupled(4, (:end, :i_forward, :j_forward), Float64),
                                       y_neg=BoundaryConditionCoupled(5, (:end, :i_forward, :j_forward), Float64),
                                       y_pos=BoundaryConditionCoupled(6, (:end, :i_forward, :j_forward), Float64),
                                       z_neg=boundary_condition,
                                       z_pos=boundary_condition,
                                     ))

mesh3 = StructuredMesh(cells_per_dimension, cubed_sphere_mapping(inner_radius, thickness, 3), periodicity=false,
                   mapping_as_string=mapping_as_string_(inner_radius, thickness, 3))

semi3 = SemidiscretizationHyperbolic(mesh3, equations, initial_condition, solver,
                                     source_terms=source_terms_convergence_test,
                                     boundary_conditions=(
                                       x_neg=BoundaryConditionCoupled(1, (:begin, :i_forward, :j_forward), Float64),
                                       x_pos=BoundaryConditionCoupled(2, (:begin, :i_forward, :j_forward), Float64),
                                       y_neg=BoundaryConditionCoupled(5, (:i_forward, :begin, :j_forward), Float64),
                                       y_pos=BoundaryConditionCoupled(6, (:i_forward, :begin, :j_forward), Float64),
                                       z_neg=boundary_condition,
                                       z_pos=boundary_condition,
                                     ))

mesh4 = StructuredMesh(cells_per_dimension, cubed_sphere_mapping(inner_radius, thickness, 4), periodicity=false,
                   mapping_as_string=mapping_as_string_(inner_radius, thickness, 4))

semi4 = SemidiscretizationHyperbolic(mesh4, equations, initial_condition, solver,
                                     source_terms=source_terms_convergence_test,
                                     boundary_conditions=(
                                       x_neg=BoundaryConditionCoupled(1, (:end, :i_forward, :j_forward), Float64),
                                       x_pos=BoundaryConditionCoupled(2, (:end, :i_forward, :j_forward), Float64),
                                       y_neg=BoundaryConditionCoupled(5, (:i_forward, :end, :j_forward), Float64),
                                       y_pos=BoundaryConditionCoupled(6, (:i_forward, :end, :j_forward), Float64),
                                       z_neg=boundary_condition,
                                       z_pos=boundary_condition,
                                     ))

mesh5 = StructuredMesh(cells_per_dimension, cubed_sphere_mapping(inner_radius, thickness, 5), periodicity=false,
                   mapping_as_string=mapping_as_string_(inner_radius, thickness, 5))

semi5 = SemidiscretizationHyperbolic(mesh5, equations, initial_condition, solver,
                                     source_terms=source_terms_convergence_test,
                                     boundary_conditions=(
                                       x_neg=BoundaryConditionCoupled(1, (:i_forward, :begin, :j_forward), Float64),
                                       x_pos=BoundaryConditionCoupled(2, (:i_forward, :begin, :j_forward), Float64),
                                       y_neg=BoundaryConditionCoupled(3, (:i_forward, :begin, :j_forward), Float64),
                                       y_pos=BoundaryConditionCoupled(4, (:i_forward, :begin, :j_forward), Float64),
                                       z_neg=boundary_condition,
                                       z_pos=boundary_condition,
                                     ))

mesh6 = StructuredMesh(cells_per_dimension, cubed_sphere_mapping(inner_radius, thickness, 6), periodicity=false,
                   mapping_as_string=mapping_as_string_(inner_radius, thickness, 6))

semi6 = SemidiscretizationHyperbolic(mesh6, equations, initial_condition, solver,
                                     source_terms=source_terms_convergence_test,
                                     boundary_conditions=(
                                       x_neg=BoundaryConditionCoupled(1, (:i_forward, :end, :j_forward), Float64),
                                       x_pos=BoundaryConditionCoupled(2, (:i_forward, :end, :j_forward), Float64),
                                       y_neg=BoundaryConditionCoupled(3, (:i_forward, :end, :j_forward), Float64),
                                       y_pos=BoundaryConditionCoupled(4, (:i_forward, :end, :j_forward), Float64),
                                       z_neg=boundary_condition,
                                       z_pos=boundary_condition,
                                     ))


semi = SemidiscretizationCoupled((semi1, semi2, semi3, semi4, semi5, semi6))


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
