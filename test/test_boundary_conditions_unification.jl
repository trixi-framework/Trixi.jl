module TestBoundaryConditionsUnification

using Test
using Trixi

include("test_trixi.jl")

# Test that boundary conditions work with both Dict and NamedTuple formats for all mesh types

@testset "Boundary conditions unification" begin
#! format: noindent

@timed_testset "TreeMesh 2D" begin
    equations = LinearScalarAdvectionEquation2D((0.2, -0.7))

    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0),
                    initial_refinement_level = 2,
                    n_cells_max = 10_000,
                    periodicity = false)

    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

    initial_condition = initial_condition_convergence_test
    boundary_condition = BoundaryConditionDirichlet(initial_condition)

    # Test with Dict
    boundary_conditions_dict = Dict(:x_neg => boundary_condition,
                                    :x_pos => boundary_condition,
                                    :y_neg => boundary_condition,
                                    :y_pos => boundary_condition)

    semi_dict = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          solver,
                                                          boundary_conditions = boundary_conditions_dict)

    # Verify boundary conditions were stored correctly
    @test semi_dict.boundary_conditions.x_neg === boundary_condition
    @test semi_dict.boundary_conditions.x_pos === boundary_condition
    @test semi_dict.boundary_conditions.y_neg === boundary_condition
    @test semi_dict.boundary_conditions.y_pos === boundary_condition

    # Test with NamedTuple
    boundary_conditions_named = (x_neg = boundary_condition,
                                 x_pos = boundary_condition,
                                 y_neg = boundary_condition,
                                 y_pos = boundary_condition)

    semi_named = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                           initial_condition,
                                                           solver,
                                                           boundary_conditions = boundary_conditions_named)

    # Verify boundary conditions were stored correctly
    @test semi_named.boundary_conditions.x_neg === boundary_condition
    @test semi_named.boundary_conditions.x_pos === boundary_condition
    @test semi_named.boundary_conditions.y_neg === boundary_condition
    @test semi_named.boundary_conditions.y_pos === boundary_condition
end

@timed_testset "StructuredMesh 2D" begin
    equations = CompressibleEulerEquations2D(1.4)

    mapping(xi, eta) = (xi, eta)

    mesh = StructuredMesh((4, 4), mapping, periodicity = false)

    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

    initial_condition = initial_condition_convergence_test
    boundary_condition = BoundaryConditionDirichlet(initial_condition)

    # Test with Dict
    boundary_conditions_dict = Dict(:x_neg => boundary_condition,
                                    :x_pos => boundary_condition,
                                    :y_neg => boundary_condition,
                                    :y_pos => boundary_condition)

    semi_dict = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          solver,
                                                          boundary_conditions = boundary_conditions_dict)

    # Verify boundary conditions were stored correctly
    @test semi_dict.boundary_conditions.x_neg === boundary_condition
    @test semi_dict.boundary_conditions.x_pos === boundary_condition
    @test semi_dict.boundary_conditions.y_neg === boundary_condition
    @test semi_dict.boundary_conditions.y_pos === boundary_condition

    # Test with NamedTuple
    boundary_conditions_named = (x_neg = boundary_condition,
                                 x_pos = boundary_condition,
                                 y_neg = boundary_condition,
                                 y_pos = boundary_condition)

    semi_named = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                           initial_condition,
                                                           solver,
                                                           boundary_conditions = boundary_conditions_named)

    # Verify boundary conditions were stored correctly
    @test semi_named.boundary_conditions.x_neg === boundary_condition
    @test semi_named.boundary_conditions.x_pos === boundary_condition
    @test semi_named.boundary_conditions.y_neg === boundary_condition
    @test semi_named.boundary_conditions.y_pos === boundary_condition
end

@timed_testset "P4estMesh" begin
    equations = LinearScalarAdvectionEquation2D((0.2, -0.7))

    trees_per_dimension = (2, 2)
    mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                     coordinates_min = (-1.0, -1.0),
                     coordinates_max = (1.0, 1.0),
                     periodicity = (false, false),
                     initial_refinement_level = 0)

    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

    initial_condition = initial_condition_convergence_test
    boundary_condition = BoundaryConditionDirichlet(initial_condition)

    # Test with NamedTuple
    boundary_conditions_named = (x_neg = boundary_condition,
                                 x_pos = boundary_condition,
                                 y_neg = boundary_condition,
                                 y_pos = boundary_condition)

    semi_named = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                           initial_condition,
                                                           solver,
                                                           boundary_conditions = boundary_conditions_named)

    # Verify boundary conditions - P4estMesh stores them in a special sorted structure
    @test semi_named.boundary_conditions isa Trixi.UnstructuredSortedBoundaryTypes

    # Test with Dict
    boundary_conditions_dict = Dict(:x_neg => boundary_condition,
                                    :x_pos => boundary_condition,
                                    :y_neg => boundary_condition,
                                    :y_pos => boundary_condition)

    semi_dict = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          solver,
                                                          boundary_conditions = boundary_conditions_dict)

    # Verify boundary conditions - P4estMesh stores them in a special sorted structure
    @test semi_dict.boundary_conditions isa Trixi.UnstructuredSortedBoundaryTypes
end

@timed_testset "DGMultiMesh" begin
    dg = DGMulti(polydeg = 3, element_type = Tri(),
                 approximation_type = Polynomial(),
                 surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
                 volume_integral = VolumeIntegralWeakForm())

    equations = CompressibleEulerEquations2D(1.4)

    initial_condition = initial_condition_convergence_test

    # Create simple boundary function
    top_boundary(x, tol = 50 * eps()) = abs(x[2] - 1) < tol
    rest_of_boundary(x, tol = 50 * eps()) = !top_boundary(x, tol)
    is_on_boundary = Dict(:top => top_boundary, :rest => rest_of_boundary)

    cells_per_dimension = (4, 4)
    mesh = DGMultiMesh(dg, cells_per_dimension, is_on_boundary = is_on_boundary)

    boundary_condition = BoundaryConditionDirichlet(initial_condition)

    # Test with Dict
    boundary_conditions_dict = Dict(:top => boundary_condition,
                                    :rest => boundary_condition)

    semi_dict = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          dg,
                                                          boundary_conditions = boundary_conditions_dict)

    # Verify boundary conditions were stored (DGMultiMesh uses NamedTuple internally)
    @test haskey(semi_dict.boundary_conditions, :top)
    @test haskey(semi_dict.boundary_conditions, :rest)
    @test semi_dict.boundary_conditions.top === boundary_condition
    @test semi_dict.boundary_conditions.rest === boundary_condition

    # Test with NamedTuple
    boundary_conditions_named = (top = boundary_condition,
                                 rest = boundary_condition)

    semi_named = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                           initial_condition,
                                                           dg,
                                                           boundary_conditions = boundary_conditions_named)

    # Verify boundary conditions were stored
    @test haskey(semi_named.boundary_conditions, :top)
    @test haskey(semi_named.boundary_conditions, :rest)
    @test semi_named.boundary_conditions.top === boundary_condition
    @test semi_named.boundary_conditions.rest === boundary_condition
end

@timed_testset "T8codeMesh 2D" begin
    equations = LinearScalarAdvectionEquation2D((0.2, -0.7))

    trees_per_dimension = (2, 2)
    mesh = T8codeMesh(trees_per_dimension, polydeg = 1,
                      coordinates_min = (-1.0, -1.0),
                      coordinates_max = (1.0, 1.0),
                      periodicity = (false, false),
                      initial_refinement_level = 0)

    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

    initial_condition = initial_condition_convergence_test
    boundary_condition = BoundaryConditionDirichlet(initial_condition)

    # Test with NamedTuple
    boundary_conditions_named = (x_neg = boundary_condition,
                                 x_pos = boundary_condition,
                                 y_neg = boundary_condition,
                                 y_pos = boundary_condition)

    semi_named = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                           initial_condition,
                                                           solver,
                                                           boundary_conditions = boundary_conditions_named)

    # Verify boundary conditions - T8codeMesh stores them in a special sorted structure
    @test semi_named.boundary_conditions isa Trixi.UnstructuredSortedBoundaryTypes

    # Test with Dict
    boundary_conditions_dict = Dict(:x_neg => boundary_condition,
                                    :x_pos => boundary_condition,
                                    :y_neg => boundary_condition,
                                    :y_pos => boundary_condition)

    semi_dict = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          solver,
                                                          boundary_conditions = boundary_conditions_dict)

    # Verify boundary conditions - T8codeMesh stores them in a special sorted structure
    @test semi_dict.boundary_conditions isa Trixi.UnstructuredSortedBoundaryTypes
end

@timed_testset "UnstructuredMesh2D" begin
    equations = LinearScalarAdvectionEquation2D((0.2, -0.7))

    # Use the unstructured mesh file from examples
    mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/52056f1487853fab63b7f4ed7f171c80/raw/9d573387dfdbb8bce2a55db7246f4207663ac07f/mesh_trixi_unstructured_mesh_docs.mesh",
                               joinpath(@__DIR__,
                                        "mesh_trixi_unstructured_mesh_docs.mesh"))

    mesh = UnstructuredMesh2D(mesh_file, periodicity = false)

    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

    initial_condition = initial_condition_convergence_test
    boundary_condition = BoundaryConditionDirichlet(initial_condition)

    # Test with NamedTuple
    boundary_conditions_named = (Bezier = boundary_condition,
                                 Slant = boundary_condition,
                                 Right = boundary_condition,
                                 Bottom = boundary_condition,
                                 Top = boundary_condition)

    semi_named = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                           initial_condition,
                                                           solver,
                                                           boundary_conditions = boundary_conditions_named)

    # Verify boundary conditions - UnstructuredMesh2D stores them in a special sorted structure
    @test semi_named.boundary_conditions isa Trixi.UnstructuredSortedBoundaryTypes

    # Test with Dict
    boundary_conditions_dict = Dict(:Bezier => boundary_condition,
                                    :Slant => boundary_condition,
                                    :Right => boundary_condition,
                                    :Bottom => boundary_condition,
                                    :Top => boundary_condition)

    semi_dict = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          solver,
                                                          boundary_conditions = boundary_conditions_dict)

    # Verify boundary conditions - UnstructuredMesh2D stores them in a special sorted structure
    @test semi_dict.boundary_conditions isa Trixi.UnstructuredSortedBoundaryTypes
end

@timed_testset "TreeMesh with partially periodic boundaries" begin
    equations = LinearScalarAdvectionEquation2D((0.2, -0.7))

    # Mesh with x non-periodic, y periodic
    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0),
                    initial_refinement_level = 2,
                    n_cells_max = 10_000,
                    periodicity = (false, true))

    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

    initial_condition = initial_condition_convergence_test
    boundary_condition = BoundaryConditionDirichlet(initial_condition)

    # Test with Dict - only specify non-periodic boundaries
    boundary_conditions_dict = Dict(:x_neg => boundary_condition,
                                    :x_pos => boundary_condition)

    semi_dict = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          solver,
                                                          boundary_conditions = boundary_conditions_dict)

    # Verify boundary conditions were stored correctly
    @test semi_dict.boundary_conditions.x_neg === boundary_condition
    @test semi_dict.boundary_conditions.x_pos === boundary_condition
    # y boundaries should be filled with periodic BCs
    @test semi_dict.boundary_conditions.y_neg === Trixi.boundary_condition_periodic
    @test semi_dict.boundary_conditions.y_pos === Trixi.boundary_condition_periodic

    # Test with NamedTuple - only specify non-periodic boundaries
    boundary_conditions_named = (x_neg = boundary_condition,
                                 x_pos = boundary_condition)

    semi_named = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                           initial_condition,
                                                           solver,
                                                           boundary_conditions = boundary_conditions_named)

    # Verify boundary conditions were stored correctly
    @test semi_named.boundary_conditions.x_neg === boundary_condition
    @test semi_named.boundary_conditions.x_pos === boundary_condition
    # y boundaries should be filled with periodic BCs
    @test semi_named.boundary_conditions.y_neg === Trixi.boundary_condition_periodic
    @test semi_named.boundary_conditions.y_pos === Trixi.boundary_condition_periodic
end

@timed_testset "StructuredMesh with partially periodic boundaries" begin
    equations = CompressibleEulerEquations2D(1.4)

    mapping(xi, eta) = (xi, eta)

    # Mesh with x periodic, y non-periodic
    mesh = StructuredMesh((4, 4), mapping, periodicity = (true, false))

    solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

    initial_condition = initial_condition_convergence_test
    boundary_condition = BoundaryConditionDirichlet(initial_condition)

    # Test with Dict - only specify non-periodic boundaries
    boundary_conditions_dict = Dict(:y_neg => boundary_condition,
                                    :y_pos => boundary_condition)

    semi_dict = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                          initial_condition,
                                                          solver,
                                                          boundary_conditions = boundary_conditions_dict)

    # Verify boundary conditions were stored correctly
    @test semi_dict.boundary_conditions.y_neg === boundary_condition
    @test semi_dict.boundary_conditions.y_pos === boundary_condition
    # x boundaries should be filled with periodic BCs
    @test semi_dict.boundary_conditions.x_neg === Trixi.boundary_condition_periodic
    @test semi_dict.boundary_conditions.x_pos === Trixi.boundary_condition_periodic

    # Test with NamedTuple - only specify non-periodic boundaries
    boundary_conditions_named = (y_neg = boundary_condition,
                                 y_pos = boundary_condition)

    semi_named = @test_nowarn SemidiscretizationHyperbolic(mesh, equations,
                                                           initial_condition,
                                                           solver,
                                                           boundary_conditions = boundary_conditions_named)

    # Verify boundary conditions were stored correctly
    @test semi_named.boundary_conditions.y_neg === boundary_condition
    @test semi_named.boundary_conditions.y_pos === boundary_condition
    # x boundaries should be filled with periodic BCs
    @test semi_named.boundary_conditions.x_neg === Trixi.boundary_condition_periodic
    @test semi_named.boundary_conditions.x_pos === Trixi.boundary_condition_periodic
end
end

end # module
