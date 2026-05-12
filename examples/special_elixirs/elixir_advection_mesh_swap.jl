# Demonstrates the unified mesh constructor interface for rectangular domains.
# The same 2D linear advection setup is run with different mesh types using
# the keyword-only constructor style with `initial_refinement_level`.
#
# See docs/src/meshes/mesh_constructor_comparison.md for more details

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
#  Parameters

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# The solution polynomial degree here is only used by the solver and independent of the mesh geometry
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 4  # 2^4 = 16 cells per dimension

t_end = 1.0

###############################################################################
# Helper function: build semi and solve for a mesh
# Based on: examples/tree_2d_dgsem/elixir_advection_basic.jl

function run_advection(mesh)
    semi = SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition_convergence_test, solver;
                                        boundary_conditions = boundary_condition_periodic)
    ode = semidiscretize(semi, (0.0, t_end))
    stepsize_callback = StepsizeCallback(cfl = 1.6)
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = 1.0, ode_default_options()...,
                callback = CallbackSet(stepsize_callback))
    return sol
end

###############################################################################
# Keyword-only interface for mesh constructors

mesh = TreeMesh(coordinates_min = coordinates_min,
                coordinates_max = coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 30_000, periodicity = true)
sol = run_advection(mesh)

mesh = StructuredMesh(coordinates_min = coordinates_min,
                      coordinates_max = coordinates_max,
                      initial_refinement_level = initial_refinement_level,
                      periodicity = true)
sol = run_advection(mesh)

# polydeg = 1 at default for P4estMesh
mesh = P4estMesh(coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 initial_refinement_level = initial_refinement_level,
                 periodicity = true)
sol = run_advection(mesh)
