# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    P4estMesh{NDIMS, NDIMS_AMBIENT} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees that uses the C library `p4est`
to manage trees and mesh refinement.

The parameter `NDIMS` denotes the dimension of the spatial domain or manifold represented
by the mesh itself, while `NDIMS_AMBIENT` denotes the dimension of the ambient space in
which the mesh is embedded. For example, the type `P4estMesh{3, 3}` corresponds to a
standard mesh for a three-dimensional volume, whereas `P4estMesh{2, 3}` corresponds to a
mesh for a two-dimensional surface or shell in three-dimensional space.

!!! warning "Experimental implementation"
    The use of `NDIMS != NDIMS_AMBIENT` is an experimental feature and may change in future
    releases.
"""
mutable struct P4estMesh{NDIMS, NDIMS_AMBIENT, RealT <: Real, IsParallel, P, Ghost,
                         NDIMSP2, NNODES} <:
               AbstractMesh{NDIMS}
    p4est       :: P # Either PointerWrapper{p4est_t} or PointerWrapper{p8est_t}
    is_parallel :: IsParallel
    ghost       :: Ghost # Either PointerWrapper{p4est_ghost_t} or PointerWrapper{p8est_ghost_t}
    # Coordinates at the nodes specified by the tensor product of `nodes` (NDIMS times).
    # This specifies the geometry interpolation for each tree.
    tree_node_coordinates::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree]
    nodes::SVector{NNODES, RealT}
    boundary_names::Array{Symbol, 2}      # [face direction, tree]
    current_filename::String
    unsaved_changes::Bool
    p4est_partition_allow_for_coarsening::Bool

    function P4estMesh{NDIMS}(p4est, tree_node_coordinates, nodes, boundary_names,
                              current_filename, unsaved_changes,
                              p4est_partition_allow_for_coarsening) where {NDIMS}
        if NDIMS == 2
            @assert p4est isa Ptr{p4est_t}
        elseif NDIMS == 3
            @assert p4est isa Ptr{p8est_t}
        end

        if mpi_isparallel()
            if !P4est.uses_mpi()
                error("p4est library does not support MPI")
            end
            is_parallel = True()
        else
            is_parallel = False()
        end

        p4est_pw = PointerWrapper(p4est)

        ghost = ghost_new_p4est(p4est)
        ghost_pw = PointerWrapper(ghost)

        # To enable the treatment of a manifold of dimension NDIMS embedded within an
        # ambient space of dimension NDIMS_AMBIENT, we store both as type parameters and
        # allow them to differ in the general case. This functionality is used for
        # constructing discretizations on spherical shell domains for applications in
        # global atmospheric modelling. The ambient dimension NDIMS_AMBIENT is therefore 
        # set here in the inner constructor to size(tree_node_coordinates, 1).
        mesh = new{NDIMS, size(tree_node_coordinates, 1),
                   eltype(tree_node_coordinates), typeof(is_parallel),
                   typeof(p4est_pw), typeof(ghost_pw), NDIMS + 2, length(nodes)}(p4est_pw,
                                                                                 is_parallel,
                                                                                 ghost_pw,
                                                                                 tree_node_coordinates,
                                                                                 nodes,
                                                                                 boundary_names,
                                                                                 current_filename,
                                                                                 unsaved_changes,
                                                                                 p4est_partition_allow_for_coarsening)

        # Destroy `p4est` structs when the mesh is garbage collected
        finalizer(destroy_mesh, mesh)

        return mesh
    end
end

const SerialP4estMesh{NDIMS} = P4estMesh{NDIMS, <:Any, <:Real, <:False}
const ParallelP4estMesh{NDIMS} = P4estMesh{NDIMS, <:Any, <:Real, <:True}

@inline mpi_parallel(mesh::SerialP4estMesh) = False()
@inline mpi_parallel(mesh::ParallelP4estMesh) = True()

function destroy_mesh(mesh::P4estMesh{2})
    connectivity = mesh.p4est.connectivity
    p4est_ghost_destroy(mesh.ghost)
    p4est_destroy(mesh.p4est)
    p4est_connectivity_destroy(connectivity)
end

function destroy_mesh(mesh::P4estMesh{3})
    connectivity = mesh.p4est.connectivity
    p8est_ghost_destroy(mesh.ghost)
    p8est_destroy(mesh.p4est)
    p8est_connectivity_destroy(connectivity)
end

@inline Base.ndims(::P4estMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMesh{NDIMS, NDIMS_AMBIENT, RealT}) where {NDIMS, NDIMS_AMBIENT, RealT} = RealT
@inline ndims_ambient(::P4estMesh{NDIMS, NDIMS_AMBIENT}) where {NDIMS, NDIMS_AMBIENT} = NDIMS_AMBIENT

@inline function ntrees(mesh::P4estMesh)
    return mesh.p4est.trees.elem_count[]
end
# returns Int32 by default which causes a weird method error when creating the cache
@inline ncells(mesh::P4estMesh) = Int(mesh.p4est.local_num_quadrants[])
@inline ncellsglobal(mesh::P4estMesh) = Int(mesh.p4est.global_num_quadrants[])

function Base.show(io::IO, mesh::P4estMesh)
    print(io, "P4estMesh{", ndims(mesh), ", ", ndims_ambient(mesh), ", ", real(mesh),
          "}")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::P4estMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        setup = [
            "#trees" => ntrees(mesh),
            "current #cells" => ncellsglobal(mesh),
            "polydeg" => length(mesh.nodes) - 1
        ]
        summary_box(io,
                    "P4estMesh{" * string(ndims(mesh)) * ", " *
                    string(ndims_ambient(mesh)) *
                    ", " * string(real(mesh)) * "}", setup)
    end
end

"""
    P4estMesh(trees_per_dimension; polydeg,
              mapping=nothing, faces=nothing, coordinates_min=nothing, coordinates_max=nothing,
              RealT=Float64, initial_refinement_level=0, periodicity=true, unsaved_changes=true,
              p4est_partition_allow_for_coarsening=true)

Create a structured curved `P4estMesh` of the specified size.

There are three ways to map the mesh to the physical domain.
1. Define a `mapping` that maps the hypercube `[-1, 1]^n`.
2. Specify a `Tuple` `faces` of functions that parametrize each face.
3. Create a rectangular mesh by specifying `coordinates_min` and `coordinates_max`.

Non-periodic boundaries will be called `:x_neg`, `:x_pos`, `:y_neg`, `:y_pos`, `:z_neg`, `:z_pos`.

# Arguments
- `trees_per_dimension::NTupleE{NDIMS, Int}`: the number of trees in each dimension.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the reference mesh (`[-1, 1]^n`) to the physical domain.
             Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `faces::NTuple{2*NDIMS}`: a tuple of `2 * NDIMS` functions that describe the faces of the domain.
                            Each function must take `NDIMS-1` arguments.
                            `faces[1]` describes the face onto which the face in negative x-direction
                            of the unit hypercube is mapped. The face in positive x-direction of
                            the unit hypercube will be mapped onto the face described by `faces[2]`.
                            `faces[3:4]` describe the faces in positive and negative y-direction respectively
                            (in 2D and 3D).
                            `faces[5:6]` describe the faces in positive and negative z-direction respectively (in 3D).
                            Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `coordinates_min`: vector or tuple of the coordinates of the corner in the negative direction of each dimension
                     to create a rectangular mesh.
                     Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `coordinates_max`: vector or tuple of the coordinates of the corner in the positive direction of each dimension
                     to create a rectangular mesh.
                     Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}`
                 deciding for each dimension if the boundaries in this dimension are periodic.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
- `p4est_partition_allow_for_coarsening::Bool`: Must be `true` when using AMR to make mesh adaptivity
                                                independent of domain partitioning. Should be `false` for static meshes
                                                to permit more fine-grained partitioning.
"""
function P4estMesh(trees_per_dimension; polydeg,
                   mapping = nothing, faces = nothing, coordinates_min = nothing,
                   coordinates_max = nothing,
                   RealT = Float64, initial_refinement_level = 0, periodicity = true,
                   unsaved_changes = true,
                   p4est_partition_allow_for_coarsening = true)
    @assert ((coordinates_min === nothing)===(coordinates_max === nothing)) "Either both or none of coordinates_min and coordinates_max must be specified"

    @assert count(i -> i !== nothing,
                  (mapping, faces, coordinates_min))==1 "Exactly one of mapping, faces and coordinates_min/max must be specified"

    # Extract mapping
    if faces !== nothing
        validate_faces(faces)
        mapping = transfinite_mapping(faces)
    elseif coordinates_min !== nothing
        mapping = coordinates2mapping(coordinates_min, coordinates_max)
    end

    NDIMS = length(trees_per_dimension)

    # Convert periodicity to a Tuple of a Bool for every dimension
    if all(periodicity)
        # Also catches case where periodicity = true
        periodicity = ntuple(_ -> true, NDIMS)
    elseif !any(periodicity)
        # Also catches case where periodicity = false
        periodicity = ntuple(_ -> false, NDIMS)
    else
        # Default case if periodicity is an iterable
        periodicity = Tuple(periodicity)
    end

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes
    tree_node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> length(nodes),
                                                           NDIMS)...,
                                                    prod(trees_per_dimension))
    calc_tree_node_coordinates!(tree_node_coordinates, nodes, mapping,
                                trees_per_dimension)

    # p4est_connectivity_new_brick has trees in Z-order, so use our own function for this
    connectivity = connectivity_structured(trees_per_dimension..., periodicity)

    p4est = new_p4est(connectivity, initial_refinement_level)

    # Non-periodic boundaries
    boundary_names = fill(Symbol("---"), 2 * NDIMS, prod(trees_per_dimension))

    structured_boundary_names!(boundary_names, trees_per_dimension, periodicity)

    return P4estMesh{NDIMS}(p4est, tree_node_coordinates, nodes,
                            boundary_names, "", unsaved_changes,
                            p4est_partition_allow_for_coarsening)
end

# 2D version
function structured_boundary_names!(boundary_names, trees_per_dimension::NTuple{2},
                                    periodicity)
    linear_indices = LinearIndices(trees_per_dimension)

    # Boundaries in x-direction
    if !periodicity[1]
        for cell_y in 1:trees_per_dimension[2]
            tree = linear_indices[1, cell_y]
            boundary_names[1, tree] = :x_neg

            tree = linear_indices[end, cell_y]
            boundary_names[2, tree] = :x_pos
        end
    end

    # Boundaries in y-direction
    if !periodicity[2]
        for cell_x in 1:trees_per_dimension[1]
            tree = linear_indices[cell_x, 1]
            boundary_names[3, tree] = :y_neg

            tree = linear_indices[cell_x, end]
            boundary_names[4, tree] = :y_pos
        end
    end
end

# 3D version
function structured_boundary_names!(boundary_names, trees_per_dimension::NTuple{3},
                                    periodicity)
    linear_indices = LinearIndices(trees_per_dimension)

    # Boundaries in x-direction
    if !periodicity[1]
        for cell_z in 1:trees_per_dimension[3], cell_y in 1:trees_per_dimension[2]
            tree = linear_indices[1, cell_y, cell_z]
            boundary_names[1, tree] = :x_neg

            tree = linear_indices[end, cell_y, cell_z]
            boundary_names[2, tree] = :x_pos
        end
    end

    # Boundaries in y-direction
    if !periodicity[2]
        for cell_z in 1:trees_per_dimension[3], cell_x in 1:trees_per_dimension[1]
            tree = linear_indices[cell_x, 1, cell_z]
            boundary_names[3, tree] = :y_neg

            tree = linear_indices[cell_x, end, cell_z]
            boundary_names[4, tree] = :y_pos
        end
    end

    # Boundaries in z-direction
    if !periodicity[3]
        for cell_y in 1:trees_per_dimension[2], cell_x in 1:trees_per_dimension[1]
            tree = linear_indices[cell_x, cell_y, 1]
            boundary_names[5, tree] = :z_neg

            tree = linear_indices[cell_x, cell_y, end]
            boundary_names[6, tree] = :z_pos
        end
    end
end

"""
    P4estMesh{NDIMS}(meshfile::String;
                     mapping=nothing, polydeg=1, RealT=Float64,
                     initial_refinement_level=0, unsaved_changes=true,
                     p4est_partition_allow_for_coarsening=true,
                     boundary_symbols = nothing)

Main mesh constructor for the `P4estMesh` that imports an unstructured, conforming
mesh from an Abaqus mesh file (`.inp`). Each element of the conforming mesh parsed
from the `meshfile` is created as a [`p4est`](https://github.com/cburstedde/p4est)
tree datatype.

To create a curved unstructured mesh `P4estMesh` two strategies are available:

- `p4est_mesh_from_hohqmesh_abaqus`: High-order, curved boundary information created by
                                     [`HOHQMesh.jl`](https://github.com/trixi-framework/HOHQMesh.jl) is
                                     available in the `meshfile`. The mesh polynomial degree `polydeg`
                                     of the boundaries is provided from the `meshfile`. The computation of
                                     the mapped tree coordinates is done with transfinite interpolation
                                     with linear blending similar to [`UnstructuredMesh2D`](@ref). Boundary name
                                     information is also parsed from the `meshfile` such that different boundary
                                     conditions can be set at each named boundary on a given tree.
- `p4est_mesh_from_standard_abaqus`: By default, with `mapping=nothing` and `polydeg=1`, this creates a
                                     straight-sided from the information parsed from the `meshfile`. If a mapping
                                     function is specified then it computes the mapped tree coordinates via polynomial
                                     interpolants with degree `polydeg`. The mesh created by this function will only
                                     have one boundary `:all` if `boundary_symbols` is not specified.
                                     If `boundary_symbols` is specified the mesh file will be parsed for nodesets defining
                                     the boundary nodes from which boundary edges (2D) and faces (3D) will be assigned.

Note that the `mapping` and `polydeg` keyword arguments are only used by the `p4est_mesh_from_standard_abaqus`
function. The `p4est_mesh_from_hohqmesh_abaqus` function obtains the mesh `polydeg` directly from the `meshfile`
and constructs the transfinite mapping internally.

The particular strategy is selected according to the header present in the `meshfile` where
the constructor checks whether or not the `meshfile` was created with
[HOHQMesh.jl](https://github.com/trixi-framework/HOHQMesh.jl).
If the Abaqus file header is not present in the `meshfile` then the `P4estMesh` is created
with the function `p4est_mesh_from_standard_abaqus`.

The default keyword argument `initial_refinement_level=0` corresponds to a forest
where the number of trees is the same as the number of elements in the original `meshfile`.
Increasing the `initial_refinement_level` allows one to uniformly refine the base mesh given
in the `meshfile` to create a forest with more trees before the simulation begins.
For example, if a two-dimensional base mesh contains 25 elements then setting
`initial_refinement_level=1` creates an initial forest of `2^2 * 25 = 100` trees.

# Arguments
- `meshfile::String`: an uncurved Abaqus mesh file that can be imported by `p4est`.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
- `p4est_partition_allow_for_coarsening::Bool`: Must be `true` when using AMR to make mesh adaptivity
                                                independent of domain partitioning. Should be `false` for static meshes
                                                to permit more fine-grained partitioning.
- `boundary_symbols::Vector{Symbol}`: A vector of symbols that correspond to the boundary names in the `meshfile`.
                                      If `nothing` is passed then all boundaries are named `:all`.
"""
function P4estMesh{NDIMS}(meshfile::String;
                          mapping = nothing, polydeg = 1, RealT = Float64,
                          initial_refinement_level = 0, unsaved_changes = true,
                          p4est_partition_allow_for_coarsening = true,
                          boundary_symbols = nothing) where {NDIMS}
    # Prevent `p4est` from crashing Julia if the file doesn't exist
    @assert isfile(meshfile)

    # Read in the Header of the meshfile to determine which constructor is appropriate
    header = open(meshfile, "r") do io
        readline(io) # *Header of the Abaqus file; discarded
        readline(io) # Readin the actual header information
    end

    # Check if the meshfile was generated using HOHQMesh
    if header == " File created by HOHQMesh"
        # Mesh curvature and boundary naming is handled with additional information available in meshfile
        p4est, tree_node_coordinates, nodes, boundary_names = p4est_mesh_from_hohqmesh_abaqus(meshfile,
                                                                                              initial_refinement_level,
                                                                                              NDIMS,
                                                                                              RealT)
    else
        # Mesh curvature is handled directly by applying the mapping keyword argument
        p4est, tree_node_coordinates, nodes, boundary_names = p4est_mesh_from_standard_abaqus(meshfile,
                                                                                              mapping,
                                                                                              polydeg,
                                                                                              initial_refinement_level,
                                                                                              NDIMS,
                                                                                              RealT,
                                                                                              boundary_symbols)
    end

    return P4estMesh{NDIMS}(p4est, tree_node_coordinates, nodes,
                            boundary_names, "", unsaved_changes,
                            p4est_partition_allow_for_coarsening)
end

# Wrapper for `p4est_connectivity_from_hohqmesh_abaqus`. The latter is used
# by `T8codeMesh`, too.
function p4est_mesh_from_hohqmesh_abaqus(meshfile, initial_refinement_level,
                                         n_dimensions, RealT)
    connectivity, tree_node_coordinates, nodes, boundary_names = p4est_connectivity_from_hohqmesh_abaqus(meshfile,
                                                                                                         initial_refinement_level,
                                                                                                         n_dimensions,
                                                                                                         RealT)

    p4est = new_p4est(connectivity, initial_refinement_level)

    return p4est, tree_node_coordinates, nodes, boundary_names
end

# Wrapper for `p4est_connectivity_from_standard_abaqus`. The latter is used
# by `T8codeMesh`, too.
function p4est_mesh_from_standard_abaqus(meshfile, mapping, polydeg,
                                         initial_refinement_level, n_dimensions, RealT,
                                         boundary_symbols)
    connectivity, tree_node_coordinates, nodes, boundary_names = p4est_connectivity_from_standard_abaqus(meshfile,
                                                                                                         mapping,
                                                                                                         polydeg,
                                                                                                         initial_refinement_level,
                                                                                                         n_dimensions,
                                                                                                         RealT,
                                                                                                         boundary_symbols)

    p4est = new_p4est(connectivity, initial_refinement_level)

    return p4est, tree_node_coordinates, nodes, boundary_names
end

# Create the mesh connectivity, mapped node coordinates within each tree, reference nodes in [-1,1]
# and a list of boundary names for the `P4estMesh`. High-order boundary curve information as well as
# the boundary names on each tree are provided by the `meshfile` created by
# [`HOHQMesh.jl`](https://github.com/trixi-framework/HOHQMesh.jl).
function p4est_connectivity_from_hohqmesh_abaqus(meshfile, initial_refinement_level,
                                                 n_dimensions, RealT)
    # Create the mesh connectivity using `p4est`
    connectivity = read_inp_p4est(meshfile, Val(n_dimensions))
    connectivity_pw = PointerWrapper(connectivity)

    # These need to be of the type Int for unsafe_wrap below to work
    n_trees::Int = connectivity_pw.num_trees[]
    n_vertices::Int = connectivity_pw.num_vertices[]

    # Extract a copy of the element vertices to compute the tree node coordinates
    vertices = unsafe_wrap(Array, connectivity_pw.vertices, (3, n_vertices))

    # Readin all the information from the mesh file into a string array
    file_lines = readlines(open(meshfile))

    # Get the file index where the mesh polynomial degree is given in the meshfile
    file_idx = findfirst(contains("** mesh polynomial degree"), file_lines)

    # Get the polynomial order of the mesh boundary information
    current_line = split(file_lines[file_idx])
    mesh_polydeg = parse(Int, current_line[6])
    mesh_nnodes = mesh_polydeg + 1

    # Create the Chebyshev-Gauss-Lobatto nodes used by HOHQMesh to represent the boundaries
    cheby_nodes, _ = chebyshev_gauss_lobatto_nodes_weights(mesh_nnodes)
    nodes = SVector{mesh_nnodes}(cheby_nodes)

    # Allocate the memory for the tree node coordinates
    tree_node_coordinates = Array{RealT, n_dimensions + 2}(undef, n_dimensions,
                                                           ntuple(_ -> length(nodes),
                                                                  n_dimensions)...,
                                                           n_trees)

    # Compute the tree node coordinates and return the updated file index
    file_idx = calc_tree_node_coordinates!(tree_node_coordinates, file_lines, nodes,
                                           vertices, RealT)

    # Allocate the memory for the boundary labels
    boundary_names = Array{Symbol}(undef, (2 * n_dimensions, n_trees))

    # Read in the boundary names from the last portion of the meshfile
    # Note here the boundary names where "---" means an internal connection
    for tree in 1:n_trees
        current_line = split(file_lines[file_idx])
        boundary_names[:, tree] = map(Symbol, current_line[2:end])
        file_idx += 1
    end

    return connectivity, tree_node_coordinates, nodes, boundary_names
end

# Create the mesh connectivity, mapped node coordinates within each tree, reference nodes in [-1,1]
# and a list of boundary names for the `P4estMesh`. The tree node coordinates are computed according to
# the `mapping` passed to this function using polynomial interpolants of degree `polydeg`. All boundary
# names are given the name `:all`.
function p4est_connectivity_from_standard_abaqus(meshfile, mapping, polydeg,
                                                 initial_refinement_level, n_dimensions,
                                                 RealT,
                                                 boundary_symbols)
    # Create the mesh connectivity using `p4est`
    connectivity = read_inp_p4est(meshfile, Val(n_dimensions))
    connectivity_pw = PointerWrapper(connectivity)

    # These need to be of the type Int for unsafe_wrap below to work
    n_trees::Int = connectivity_pw.num_trees[]
    n_vertices::Int = connectivity_pw.num_vertices[]

    vertices = unsafe_wrap(Array, connectivity_pw.vertices, (3, n_vertices))
    tree_to_vertex = unsafe_wrap(Array, connectivity_pw.tree_to_vertex,
                                 (2^n_dimensions, n_trees))

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes

    tree_node_coordinates = Array{RealT, n_dimensions + 2}(undef, n_dimensions,
                                                           ntuple(_ -> length(nodes),
                                                                  n_dimensions)...,
                                                           n_trees)
    calc_tree_node_coordinates!(tree_node_coordinates, nodes, mapping, vertices,
                                tree_to_vertex)

    if boundary_symbols === nothing
        # There's no simple and generic way to distinguish boundaries without any information given.
        # Name all of them :all.
        boundary_names = fill(:all, 2 * n_dimensions, n_trees)
    else # Boundary information given
        # Read in nodes belonging to boundaries
        node_set_dict = parse_node_sets(meshfile, boundary_symbols)
        # Read in all elements with associated nodes to specify the boundaries
        element_node_matrix = parse_elements(meshfile, n_trees, n_dimensions)

        # Initialize boundary information matrix with symbol for no boundary / internal connection
        boundary_names = fill(Symbol("---"), 2 * n_dimensions, n_trees)

        # Fill `boundary_names` such that it can be processed by p4est
        assign_boundaries_standard_abaqus!(boundary_names, n_trees,
                                           element_node_matrix, node_set_dict,
                                           Val(n_dimensions))
    end

    return connectivity, tree_node_coordinates, nodes, boundary_names
end

function parse_elements(meshfile, n_trees, n_dims)
    @assert n_dims in (2, 3) "Only 2D and 3D meshes are supported"
    # Valid element types (that can be processed by p4est) based on dimension
    element_types = n_dims == 2 ?
                    ["*ELEMENT, type=CPS4", "*ELEMENT, type=C2D4",
        "*ELEMENT, type=S4"] : ["*ELEMENT, type=C3D8"]
    # 2D quads: 4 nodes + element index, 3D hexes: 8 nodes + element index
    expected_content_length = n_dims == 2 ? 5 : 9

    element_node_matrix = Matrix{Int64}(undef, n_trees, expected_content_length - 1)
    el_list_follows = false
    tree_id = 1

    open(meshfile, "r") do file
        for line in eachline(file)
            if any(startswith(line, el_type) for el_type in element_types)
                el_list_follows = true
            elseif el_list_follows
                content = split(line, ",")
                if length(content) == expected_content_length # Check that we still read in connectivity data
                    content_int = parse.(Int64, content)
                    # Add constituent nodes to the element_node_matrix.
                    # Important: Do not use index from the Abaqus file, but the one from p4est.
                    element_node_matrix[tree_id, :] = content_int[2:end] # First entry is element id
                    tree_id += 1
                else # Processed all elements for this ELSET
                    el_list_follows = false
                end
            end
        end
    end

    return element_node_matrix
end

function parse_node_sets(meshfile, boundary_symbols)
    nodes_dict = Dict{Symbol, Vector{Int64}}()
    current_symbol = nothing
    current_nodes = Int64[]

    open(meshfile, "r") do file
        for line in eachline(file)
            # Check if the line contains nodes assembled in a special set, i.e., a physical boundary
            if startswith(line, "*NSET,NSET=")
                # Safe the previous nodeset
                if current_symbol !== nothing
                    nodes_dict[current_symbol] = current_nodes
                end

                current_symbol = Symbol(split(line, "=")[2])
                if current_symbol in boundary_symbols
                    # New nodeset
                    current_nodes = Int64[]
                else # Read only boundary node sets
                    current_symbol = nothing
                end
            elseif current_symbol !== nothing # Read only if there was already a nodeset specified
                try # Check if line contains nodes
                    # There is always a trailing comma, remove the corresponding empty string
                    append!(current_nodes, parse.(Int64, split(line, ",")[1:(end - 1)]))
                catch # Something different, stop reading in nodes
                    # If parsing fails, set current_symbol to nothing
                    nodes_dict[current_symbol] = current_nodes
                    current_symbol = nothing
                end
            end
        end
        # Safe the previous nodeset
        if current_symbol !== nothing
            nodes_dict[current_symbol] = current_nodes
        end
    end

    for symbol in boundary_symbols
        if !haskey(nodes_dict, symbol)
            @warn "No nodes found for nodeset :" * "$symbol" * " !"
        end
    end

    return nodes_dict
end

# This function assigns the edges of elements to boundaries by
# checking if the nodes that define the edges are part of nodesets which correspond to boundaries.
function assign_boundaries_standard_abaqus!(boundary_names, n_trees,
                                            element_node_matrix, node_set_dict,
                                            ::Val{2}) # 2D version
    for tree in 1:n_trees
        tree_nodes = element_node_matrix[tree, :]
        # For node labeling, see
        # https://docs.software.vt.edu/abaqusv2022/English/SIMACAEELMRefMap/simaelm-r-2delem.htm#simaelm-r-2delem-t-nodedef1
        # and search for "Node ordering and face numbering on elements"
        for boundary in keys(node_set_dict) # Loop over specified boundaries
            # Check bottom edge
            if tree_nodes[1] in node_set_dict[boundary] &&
               tree_nodes[2] in node_set_dict[boundary]
                # Bottom boundary is position 3 in p4est indexing
                boundary_names[3, tree] = boundary
            end
            # Check right edge
            if tree_nodes[2] in node_set_dict[boundary] &&
               tree_nodes[3] in node_set_dict[boundary]
                # Right boundary is position 2 in p4est indexing
                boundary_names[2, tree] = boundary
            end
            # Check top edge
            if tree_nodes[3] in node_set_dict[boundary] &&
               tree_nodes[4] in node_set_dict[boundary]
                # Top boundary is position 4 in p4est indexing
                boundary_names[4, tree] = boundary
            end
            # Check left edge
            if tree_nodes[4] in node_set_dict[boundary] &&
               tree_nodes[1] in node_set_dict[boundary]
                # Left boundary is position 1 in p4est indexing
                boundary_names[1, tree] = boundary
            end
        end
    end

    return boundary_names
end

# This function assigns the edges of elements to boundaries by
# checking if the nodes that define the faces are part of nodesets which correspond to boundaries.
function assign_boundaries_standard_abaqus!(boundary_names, n_trees,
                                            element_node_matrix, node_set_dict,
                                            ::Val{3}) # 3D version
    for tree in 1:n_trees
        tree_nodes = element_node_matrix[tree, :]
        # For node labeling, see
        # https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node26.html
        for boundary in keys(node_set_dict) # Loop over specified boundaries
            # Check "front face" (y_min)
            if tree_nodes[1] in node_set_dict[boundary] &&
               tree_nodes[2] in node_set_dict[boundary] &&
               tree_nodes[5] in node_set_dict[boundary] &&
               tree_nodes[6] in node_set_dict[boundary]
                # Front face is position 3 in p4est indexing
                boundary_names[3, tree] = boundary
            end
            # Check "back face" (y_max)
            if tree_nodes[3] in node_set_dict[boundary] &&
               tree_nodes[4] in node_set_dict[boundary] &&
               tree_nodes[7] in node_set_dict[boundary] &&
               tree_nodes[8] in node_set_dict[boundary]
                # Front face is position 4 in p4est indexing
                boundary_names[4, tree] = boundary
            end
            # Check "left face" (x_min)
            if tree_nodes[1] in node_set_dict[boundary] &&
               tree_nodes[4] in node_set_dict[boundary] &&
               tree_nodes[5] in node_set_dict[boundary] &&
               tree_nodes[8] in node_set_dict[boundary]
                # Left face is position 1 in p4est indexing
                boundary_names[1, tree] = boundary
            end
            # Check "right face" (x_max)
            if tree_nodes[2] in node_set_dict[boundary] &&
               tree_nodes[3] in node_set_dict[boundary] &&
               tree_nodes[6] in node_set_dict[boundary] &&
               tree_nodes[7] in node_set_dict[boundary]
                # Right face is position 2 in p4est indexing
                boundary_names[2, tree] = boundary
            end
            # Check "bottom face" (z_min)
            if tree_nodes[1] in node_set_dict[boundary] &&
               tree_nodes[2] in node_set_dict[boundary] &&
               tree_nodes[3] in node_set_dict[boundary] &&
               tree_nodes[4] in node_set_dict[boundary]
                # Bottom face is position 5 in p4est indexing
                boundary_names[5, tree] = boundary
            end
            # Check "top face" (z_max)
            if tree_nodes[5] in node_set_dict[boundary] &&
               tree_nodes[6] in node_set_dict[boundary] &&
               tree_nodes[7] in node_set_dict[boundary] &&
               tree_nodes[8] in node_set_dict[boundary]
                # Top face is position 6 in p4est indexing
                boundary_names[6, tree] = boundary
            end
        end
    end

    return boundary_names
end

"""
    P4estMeshCubedSphere(trees_per_face_dimension, layers, inner_radius, thickness;
                         polydeg, RealT=Float64,
                         initial_refinement_level=0, unsaved_changes=true,
                         p4est_partition_allow_for_coarsening=true)

Build a "Cubed Sphere" mesh as `P4estMesh` with
`6 * trees_per_face_dimension^2 * layers` trees.

The mesh will have two boundaries, `:inside` and `:outside`.

# Arguments
- `trees_per_face_dimension::Integer`: the number of trees in the first two local dimensions of
                                       each face.
- `layers::Integer`: the number of trees in the third local dimension of each face, i.e., the number
                     of layers of the sphere.
- `inner_radius::Integer`: the inner radius of the sphere.
- `thickness::Integer`: the thickness of the sphere. The outer radius will be `inner_radius + thickness`.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
- `p4est_partition_allow_for_coarsening::Bool`: Must be `true` when using AMR to make mesh adaptivity
                                                independent of domain partitioning. Should be `false` for static meshes
                                                to permit more fine-grained partitioning.
"""
function P4estMeshCubedSphere(trees_per_face_dimension, layers, inner_radius, thickness;
                              polydeg, RealT = Float64,
                              initial_refinement_level = 0, unsaved_changes = true,
                              p4est_partition_allow_for_coarsening = true)
    connectivity = connectivity_cubed_sphere(trees_per_face_dimension, layers)

    n_trees = 6 * trees_per_face_dimension^2 * layers

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes

    tree_node_coordinates = Array{RealT, 5}(undef, 3,
                                            ntuple(_ -> length(nodes), 3)...,
                                            n_trees)
    calc_tree_node_coordinates!(tree_node_coordinates, nodes, trees_per_face_dimension,
                                layers,
                                inner_radius, thickness)

    p4est = new_p4est(connectivity, initial_refinement_level)

    boundary_names = fill(Symbol("---"), 2 * 3, n_trees)
    boundary_names[5, :] .= Symbol("inside")
    boundary_names[6, :] .= Symbol("outside")

    return P4estMesh{3}(p4est, tree_node_coordinates, nodes,
                        boundary_names, "", unsaved_changes,
                        p4est_partition_allow_for_coarsening)
end

# Create a new p4est_connectivity that represents a structured rectangle.
# Similar to p4est_connectivity_new_brick, but doesn't use Morton order.
# This order makes `calc_tree_node_coordinates!` below and the calculation
# of `boundary_names` above easier but is irrelevant otherwise.
# 2D version
function connectivity_structured(n_cells_x, n_cells_y, periodicity)
    linear_indices = LinearIndices((n_cells_x, n_cells_y))

    # Vertices represent the coordinates of the forest. This is used by `p4est`
    # to write VTK files.
    # Trixi.jl doesn't use the coordinates from `p4est`, so the vertices can be empty.
    n_vertices = 0
    n_trees = n_cells_x * n_cells_y
    # No corner connectivity is needed
    n_corners = 0
    vertices = C_NULL
    tree_to_vertex = C_NULL

    tree_to_tree = Array{p4est_topidx_t, 2}(undef, 4, n_trees)
    tree_to_face = Array{Int8, 2}(undef, 4, n_trees)

    for cell_y in 1:n_cells_y, cell_x in 1:n_cells_x
        tree = linear_indices[cell_x, cell_y]

        # Subtract 1 because `p4est` uses zero-based indexing
        # Negative x-direction
        if cell_x > 1
            tree_to_tree[1, tree] = linear_indices[cell_x - 1, cell_y] - 1
            tree_to_face[1, tree] = 1
        elseif periodicity[1]
            tree_to_tree[1, tree] = linear_indices[n_cells_x, cell_y] - 1
            tree_to_face[1, tree] = 1
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[1, tree] = tree - 1
            tree_to_face[1, tree] = 0
        end

        # Positive x-direction
        if cell_x < n_cells_x
            tree_to_tree[2, tree] = linear_indices[cell_x + 1, cell_y] - 1
            tree_to_face[2, tree] = 0
        elseif periodicity[1]
            tree_to_tree[2, tree] = linear_indices[1, cell_y] - 1
            tree_to_face[2, tree] = 0
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[2, tree] = tree - 1
            tree_to_face[2, tree] = 1
        end

        # Negative y-direction
        if cell_y > 1
            tree_to_tree[3, tree] = linear_indices[cell_x, cell_y - 1] - 1
            tree_to_face[3, tree] = 3
        elseif periodicity[2]
            tree_to_tree[3, tree] = linear_indices[cell_x, n_cells_y] - 1
            tree_to_face[3, tree] = 3
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[3, tree] = tree - 1
            tree_to_face[3, tree] = 2
        end

        # Positive y-direction
        if cell_y < n_cells_y
            tree_to_tree[4, tree] = linear_indices[cell_x, cell_y + 1] - 1
            tree_to_face[4, tree] = 2
        elseif periodicity[2]
            tree_to_tree[4, tree] = linear_indices[cell_x, 1] - 1
            tree_to_face[4, tree] = 2
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[4, tree] = tree - 1
            tree_to_face[4, tree] = 3
        end
    end

    tree_to_corner = C_NULL
    # `p4est` docs: "in trivial cases it is just a pointer to a p4est_topix value of 0."
    # We don't need corner connectivity, so this is a trivial case.
    ctt_offset = zeros(p4est_topidx_t, 1)

    corner_to_tree = C_NULL
    corner_to_corner = C_NULL

    connectivity = p4est_connectivity_new_copy(n_vertices, n_trees, n_corners,
                                               vertices, tree_to_vertex,
                                               tree_to_tree, tree_to_face,
                                               tree_to_corner, ctt_offset,
                                               corner_to_tree, corner_to_corner)

    @assert p4est_connectivity_is_valid(connectivity) == 1

    return connectivity
end

# 3D version
function connectivity_structured(n_cells_x, n_cells_y, n_cells_z, periodicity)
    linear_indices = LinearIndices((n_cells_x, n_cells_y, n_cells_z))

    # Vertices represent the coordinates of the forest. This is used by `p4est`
    # to write VTK files.
    # Trixi.jl doesn't use the coordinates from `p4est`, so the vertices can be empty.
    n_vertices = 0
    n_trees = n_cells_x * n_cells_y * n_cells_z
    # No edge connectivity is needed
    n_edges = 0
    # No corner connectivity is needed
    n_corners = 0
    vertices = C_NULL
    tree_to_vertex = C_NULL

    tree_to_tree = Array{p4est_topidx_t, 2}(undef, 6, n_trees)
    tree_to_face = Array{Int8, 2}(undef, 6, n_trees)

    for cell_z in 1:n_cells_z, cell_y in 1:n_cells_y, cell_x in 1:n_cells_x
        tree = linear_indices[cell_x, cell_y, cell_z]

        # Subtract 1 because `p4est` uses zero-based indexing
        # Negative x-direction
        if cell_x > 1
            tree_to_tree[1, tree] = linear_indices[cell_x - 1, cell_y, cell_z] - 1
            tree_to_face[1, tree] = 1
        elseif periodicity[1]
            tree_to_tree[1, tree] = linear_indices[n_cells_x, cell_y, cell_z] - 1
            tree_to_face[1, tree] = 1
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[1, tree] = tree - 1
            tree_to_face[1, tree] = 0
        end

        # Positive x-direction
        if cell_x < n_cells_x
            tree_to_tree[2, tree] = linear_indices[cell_x + 1, cell_y, cell_z] - 1
            tree_to_face[2, tree] = 0
        elseif periodicity[1]
            tree_to_tree[2, tree] = linear_indices[1, cell_y, cell_z] - 1
            tree_to_face[2, tree] = 0
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[2, tree] = tree - 1
            tree_to_face[2, tree] = 1
        end

        # Negative y-direction
        if cell_y > 1
            tree_to_tree[3, tree] = linear_indices[cell_x, cell_y - 1, cell_z] - 1
            tree_to_face[3, tree] = 3
        elseif periodicity[2]
            tree_to_tree[3, tree] = linear_indices[cell_x, n_cells_y, cell_z] - 1
            tree_to_face[3, tree] = 3
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[3, tree] = tree - 1
            tree_to_face[3, tree] = 2
        end

        # Positive y-direction
        if cell_y < n_cells_y
            tree_to_tree[4, tree] = linear_indices[cell_x, cell_y + 1, cell_z] - 1
            tree_to_face[4, tree] = 2
        elseif periodicity[2]
            tree_to_tree[4, tree] = linear_indices[cell_x, 1, cell_z] - 1
            tree_to_face[4, tree] = 2
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[4, tree] = tree - 1
            tree_to_face[4, tree] = 3
        end

        # Negative z-direction
        if cell_z > 1
            tree_to_tree[5, tree] = linear_indices[cell_x, cell_y, cell_z - 1] - 1
            tree_to_face[5, tree] = 5
        elseif periodicity[3]
            tree_to_tree[5, tree] = linear_indices[cell_x, cell_y, n_cells_z] - 1
            tree_to_face[5, tree] = 5
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[5, tree] = tree - 1
            tree_to_face[5, tree] = 4
        end

        # Positive z-direction
        if cell_z < n_cells_z
            tree_to_tree[6, tree] = linear_indices[cell_x, cell_y, cell_z + 1] - 1
            tree_to_face[6, tree] = 4
        elseif periodicity[3]
            tree_to_tree[6, tree] = linear_indices[cell_x, cell_y, 1] - 1
            tree_to_face[6, tree] = 4
        else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
            tree_to_tree[6, tree] = tree - 1
            tree_to_face[6, tree] = 5
        end
    end

    tree_to_edge = C_NULL
    # `p4est` docs: "in trivial cases it is just a pointer to a p4est_topix value of 0."
    # We don't need edge connectivity, so this is a trivial case.
    ett_offset = zeros(p4est_topidx_t, 1)
    edge_to_tree = C_NULL
    edge_to_edge = C_NULL

    tree_to_corner = C_NULL
    # `p4est` docs: "in trivial cases it is just a pointer to a p4est_topix value of 0."
    # We don't need corner connectivity, so this is a trivial case.
    ctt_offset = zeros(p4est_topidx_t, 1)

    corner_to_tree = C_NULL
    corner_to_corner = C_NULL

    connectivity = p8est_connectivity_new_copy(n_vertices, n_trees, n_corners, n_edges,
                                               vertices, tree_to_vertex,
                                               tree_to_tree, tree_to_face,
                                               tree_to_edge, ett_offset,
                                               edge_to_tree, edge_to_edge,
                                               tree_to_corner, ctt_offset,
                                               corner_to_tree, corner_to_corner)

    @assert p8est_connectivity_is_valid(connectivity) == 1

    return connectivity
end

function connectivity_cubed_sphere(trees_per_face_dimension, layers)
    n_cells_x = n_cells_y = trees_per_face_dimension
    n_cells_z = layers

    linear_indices = LinearIndices((trees_per_face_dimension, trees_per_face_dimension,
                                    layers, 6))

    # Vertices represent the coordinates of the forest. This is used by `p4est`
    # to write VTK files.
    # Trixi.jl doesn't use the coordinates from `p4est`, so the vertices can be empty.
    n_vertices = 0
    n_trees = 6 * n_cells_x * n_cells_y * n_cells_z
    # No edge connectivity is needed
    n_edges = 0
    # No corner connectivity is needed
    n_corners = 0
    vertices = C_NULL
    tree_to_vertex = C_NULL

    tree_to_tree = Array{p4est_topidx_t, 2}(undef, 6, n_trees)
    tree_to_face = Array{Int8, 2}(undef, 6, n_trees)

    # Illustration of the local coordinates of each face. ξ and η are the first
    # local coordinates of each face. The third local coordinate ζ is always
    # pointing outwards, which yields a right-handed coordinate system for each face.
    #               ┌────────────────────────────────────────────────────┐
    #              ╱│                                                   ╱│
    #             ╱ │                       ξ <───┐                    ╱ │
    #            ╱  │                            ╱                    ╱  │
    #           ╱   │                4 (+y)     V                    ╱   │
    #          ╱    │                          η                    ╱    │
    #         ╱     │                                              ╱     │
    #        ╱      │                                             ╱      │
    #       ╱       │                                            ╱       │
    #      ╱        │                                           ╱        │
    #     ╱         │                    5 (-z)   η            ╱         │
    #    ╱          │                             ↑           ╱          │
    #   ╱           │                             │          ╱           │
    #  ╱            │                       ξ <───┘         ╱            │
    # ┌────────────────────────────────────────────────────┐    2 (+x)   │
    # │             │                                      │             │
    # │             │                                      │      ξ      │
    # │             │                                      │      ↑      │
    # │    1 (-x)   │                                      │      │      │
    # │             │                                      │      │      │
    # │     ╱│      │                                      │     ╱       │
    # │    V │      │                                      │    V        │
    # │   η  ↓      │                                      │   η         │
    # │      ξ      └──────────────────────────────────────│─────────────┘
    # │            ╱         η   6 (+z)                    │            ╱
    # │           ╱          ↑                             │           ╱
    # │          ╱           │                             │          ╱
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
    for direction in 1:6
        for cell_z in 1:n_cells_z, cell_y in 1:n_cells_y, cell_x in 1:n_cells_x
            tree = linear_indices[cell_x, cell_y, cell_z, direction]

            # Subtract 1 because `p4est` uses zero-based indexing
            # Negative x-direction
            if cell_x > 1 # Connect to tree at the same face
                tree_to_tree[1, tree] = linear_indices[cell_x - 1, cell_y, cell_z,
                                                       direction] - 1
                tree_to_face[1, tree] = 1
            elseif direction == 1 # This is the -x face
                target = 4
                tree_to_tree[1, tree] = linear_indices[end, cell_y, cell_z, target] - 1
                tree_to_face[1, tree] = 1
            elseif direction == 2 # This is the +x face
                target = 3
                tree_to_tree[1, tree] = linear_indices[end, cell_y, cell_z, target] - 1
                tree_to_face[1, tree] = 1
            elseif direction == 3 # This is the -y face
                target = 1
                tree_to_tree[1, tree] = linear_indices[end, cell_y, cell_z, target] - 1
                tree_to_face[1, tree] = 1
            elseif direction == 4 # This is the +y face
                target = 2
                tree_to_tree[1, tree] = linear_indices[end, cell_y, cell_z, target] - 1
                tree_to_face[1, tree] = 1
            elseif direction == 5 # This is the -z face
                target = 2
                tree_to_tree[1, tree] = linear_indices[cell_y, 1, cell_z, target] - 1
                tree_to_face[1, tree] = 2
            else # direction == 6, this is the +z face
                target = 1
                tree_to_tree[1, tree] = linear_indices[end - cell_y + 1, end, cell_z,
                                                       target] - 1
                tree_to_face[1, tree] = 9 # first face dimensions are oppositely oriented, add 6
            end

            # Positive x-direction
            if cell_x < n_cells_x # Connect to tree at the same face
                tree_to_tree[2, tree] = linear_indices[cell_x + 1, cell_y, cell_z,
                                                       direction] - 1
                tree_to_face[2, tree] = 0
            elseif direction == 1 # This is the -x face
                target = 3
                tree_to_tree[2, tree] = linear_indices[1, cell_y, cell_z, target] - 1
                tree_to_face[2, tree] = 0
            elseif direction == 2 # This is the +x face
                target = 4
                tree_to_tree[2, tree] = linear_indices[1, cell_y, cell_z, target] - 1
                tree_to_face[2, tree] = 0
            elseif direction == 3 # This is the -y face
                target = 2
                tree_to_tree[2, tree] = linear_indices[1, cell_y, cell_z, target] - 1
                tree_to_face[2, tree] = 0
            elseif direction == 4 # This is the +y face
                target = 1
                tree_to_tree[2, tree] = linear_indices[1, cell_y, cell_z, target] - 1
                tree_to_face[2, tree] = 0
            elseif direction == 5 # This is the -z face
                target = 1
                tree_to_tree[2, tree] = linear_indices[end - cell_y + 1, 1, cell_z,
                                                       target] - 1
                tree_to_face[2, tree] = 8 # first face dimensions are oppositely oriented, add 6
            else # direction == 6, this is the +z face
                target = 2
                tree_to_tree[2, tree] = linear_indices[cell_y, end, cell_z, target] - 1
                tree_to_face[2, tree] = 3
            end

            # Negative y-direction
            if cell_y > 1 # Connect to tree at the same face
                tree_to_tree[3, tree] = linear_indices[cell_x, cell_y - 1, cell_z,
                                                       direction] - 1
                tree_to_face[3, tree] = 3
            elseif direction == 1
                target = 5
                tree_to_tree[3, tree] = linear_indices[end, end - cell_x + 1, cell_z,
                                                       target] - 1
                tree_to_face[3, tree] = 7 # first face dimensions are oppositely oriented, add 6
            elseif direction == 2
                target = 5
                tree_to_tree[3, tree] = linear_indices[1, cell_x, cell_z, target] - 1
                tree_to_face[3, tree] = 0
            elseif direction == 3
                target = 5
                tree_to_tree[3, tree] = linear_indices[end - cell_x + 1, 1, cell_z,
                                                       target] - 1
                tree_to_face[3, tree] = 8 # first face dimensions are oppositely oriented, add 6
            elseif direction == 4
                target = 5
                tree_to_tree[3, tree] = linear_indices[cell_x, end, cell_z, target] - 1
                tree_to_face[3, tree] = 3
            elseif direction == 5
                target = 3
                tree_to_tree[3, tree] = linear_indices[end - cell_x + 1, 1, cell_z,
                                                       target] - 1
                tree_to_face[3, tree] = 8 # first face dimensions are oppositely oriented, add 6
            else # direction == 6
                target = 3
                tree_to_tree[3, tree] = linear_indices[cell_x, end, cell_z, target] - 1
                tree_to_face[3, tree] = 3
            end

            # Positive y-direction
            if cell_y < n_cells_y # Connect to tree at the same face
                tree_to_tree[4, tree] = linear_indices[cell_x, cell_y + 1, cell_z,
                                                       direction] - 1
                tree_to_face[4, tree] = 2
            elseif direction == 1
                target = 6
                tree_to_tree[4, tree] = linear_indices[1, end - cell_x + 1, cell_z,
                                                       target] - 1
                tree_to_face[4, tree] = 6 # first face dimensions are oppositely oriented, add 6
            elseif direction == 2
                target = 6
                tree_to_tree[4, tree] = linear_indices[end, cell_x, cell_z, target] - 1
                tree_to_face[4, tree] = 1
            elseif direction == 3
                target = 6
                tree_to_tree[4, tree] = linear_indices[cell_x, 1, cell_z, target] - 1
                tree_to_face[4, tree] = 2
            elseif direction == 4
                target = 6
                tree_to_tree[4, tree] = linear_indices[end - cell_x + 1, end, cell_z,
                                                       target] - 1
                tree_to_face[4, tree] = 9 # first face dimensions are oppositely oriented, add 6
            elseif direction == 5
                target = 4
                tree_to_tree[4, tree] = linear_indices[cell_x, 1, cell_z, target] - 1
                tree_to_face[4, tree] = 2
            else # direction == 6
                target = 4
                tree_to_tree[4, tree] = linear_indices[end - cell_x + 1, end, cell_z,
                                                       target] - 1
                tree_to_face[4, tree] = 9 # first face dimensions are oppositely oriented, add 6
            end

            # Negative z-direction
            if cell_z > 1
                tree_to_tree[5, tree] = linear_indices[cell_x, cell_y, cell_z - 1,
                                                       direction] - 1
                tree_to_face[5, tree] = 5
            else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
                tree_to_tree[5, tree] = tree - 1
                tree_to_face[5, tree] = 4
            end

            # Positive z-direction
            if cell_z < n_cells_z
                tree_to_tree[6, tree] = linear_indices[cell_x, cell_y, cell_z + 1,
                                                       direction] - 1
                tree_to_face[6, tree] = 4
            else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
                tree_to_tree[6, tree] = tree - 1
                tree_to_face[6, tree] = 5
            end
        end
    end

    tree_to_edge = C_NULL
    # `p4est` docs: "in trivial cases it is just a pointer to a p4est_topix value of 0."
    # We don't need edge connectivity, so this is a trivial case.
    ett_offset = zeros(p4est_topidx_t, 1)
    edge_to_tree = C_NULL
    edge_to_edge = C_NULL

    tree_to_corner = C_NULL
    # `p4est` docs: "in trivial cases it is just a pointer to a p4est_topix value of 0."
    # We don't need corner connectivity, so this is a trivial case.
    ctt_offset = zeros(p4est_topidx_t, 1)

    corner_to_tree = C_NULL
    corner_to_corner = C_NULL

    connectivity = p8est_connectivity_new_copy(n_vertices, n_trees, n_corners, n_edges,
                                               vertices, tree_to_vertex,
                                               tree_to_tree, tree_to_face,
                                               tree_to_edge, ett_offset,
                                               edge_to_tree, edge_to_edge,
                                               tree_to_corner, ctt_offset,
                                               corner_to_tree, corner_to_corner)

    @assert p8est_connectivity_is_valid(connectivity) == 1

    return connectivity
end

# Calculate physical coordinates of each node of a structured mesh.
# This function assumes a structured mesh with trees in row order.
# 2D version
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4},
                                     nodes, mapping, trees_per_dimension)
    linear_indices = LinearIndices(trees_per_dimension)

    # Get cell length in reference mesh
    dx = 2 / trees_per_dimension[1]
    dy = 2 / trees_per_dimension[2]

    for cell_y in 1:trees_per_dimension[2], cell_x in 1:trees_per_dimension[1]
        tree_id = linear_indices[cell_x, cell_y]

        # Calculate node coordinates of reference mesh
        cell_x_offset = -1 + (cell_x - 1) * dx + dx / 2
        cell_y_offset = -1 + (cell_y - 1) * dy + dy / 2

        for j in eachindex(nodes), i in eachindex(nodes)
            # node_coordinates are the mapped reference node coordinates
            node_coordinates[:, i, j, tree_id] .= mapping(cell_x_offset +
                                                          dx / 2 * nodes[i],
                                                          cell_y_offset +
                                                          dy / 2 * nodes[j])
        end
    end
end

# 3D version
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{<:Any, 5},
                                     nodes, mapping, trees_per_dimension)
    linear_indices = LinearIndices(trees_per_dimension)

    # Get cell length in reference mesh
    dx = 2 / trees_per_dimension[1]
    dy = 2 / trees_per_dimension[2]
    dz = 2 / trees_per_dimension[3]

    for cell_z in 1:trees_per_dimension[3],
        cell_y in 1:trees_per_dimension[2],
        cell_x in 1:trees_per_dimension[1]

        tree_id = linear_indices[cell_x, cell_y, cell_z]

        # Calculate node coordinates of reference mesh
        cell_x_offset = -1 + (cell_x - 1) * dx + dx / 2
        cell_y_offset = -1 + (cell_y - 1) * dy + dy / 2
        cell_z_offset = -1 + (cell_z - 1) * dz + dz / 2

        for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
            # node_coordinates are the mapped reference node coordinates
            node_coordinates[:, i, j, k, tree_id] .= mapping(cell_x_offset +
                                                             dx / 2 * nodes[i],
                                                             cell_y_offset +
                                                             dy / 2 * nodes[j],
                                                             cell_z_offset +
                                                             dz / 2 * nodes[k])
        end
    end
end

# Calculate physical coordinates of each node of an unstructured mesh.
# Extract corners of each tree from the connectivity,
# interpolate to requested interpolation nodes,
# map the resulting coordinates with the specified mapping.
# 2D version
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{RealT, 4},
                                     nodes, mapping,
                                     vertices, tree_to_vertex) where {RealT}
    nodes_in = [-1.0, 1.0]
    matrix = polynomial_interpolation_matrix(nodes_in, nodes)
    data_in = Array{RealT, 3}(undef, 2, 2, 2)
    tmp1 = zeros(RealT, 2, length(nodes), length(nodes_in))

    for tree in 1:size(tree_to_vertex, 2)
        # Tree vertices are stored in Z-order, ignore z-coordinate in 2D, zero-based indexing
        @views data_in[:, 1, 1] .= vertices[1:2, tree_to_vertex[1, tree] + 1]
        @views data_in[:, 2, 1] .= vertices[1:2, tree_to_vertex[2, tree] + 1]
        @views data_in[:, 1, 2] .= vertices[1:2, tree_to_vertex[3, tree] + 1]
        @views data_in[:, 2, 2] .= vertices[1:2, tree_to_vertex[4, tree] + 1]

        # Interpolate corner coordinates to specified nodes
        multiply_dimensionwise!(view(node_coordinates, :, :, :, tree),
                                matrix, matrix,
                                data_in,
                                tmp1)
    end

    map_node_coordinates!(node_coordinates, mapping)
end

function map_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4}, mapping)
    for tree in axes(node_coordinates, 4),
        j in axes(node_coordinates, 3),
        i in axes(node_coordinates, 2)

        node_coordinates[:, i, j, tree] .= mapping(node_coordinates[1, i, j, tree],
                                                   node_coordinates[2, i, j, tree])
    end

    return node_coordinates
end

function map_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4},
                               mapping::Nothing)
    return node_coordinates
end

# 3D version
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{RealT, 5},
                                     nodes, mapping,
                                     vertices, tree_to_vertex) where {RealT}
    nodes_in = [-1.0, 1.0]
    matrix = polynomial_interpolation_matrix(nodes_in, nodes)
    data_in = Array{RealT, 4}(undef, 3, 2, 2, 2)

    for tree in 1:size(tree_to_vertex, 2)
        # Tree vertices are stored in Z-order, zero-based indexing
        @views data_in[:, 1, 1, 1] .= vertices[:, tree_to_vertex[1, tree] + 1]
        @views data_in[:, 2, 1, 1] .= vertices[:, tree_to_vertex[2, tree] + 1]
        @views data_in[:, 1, 2, 1] .= vertices[:, tree_to_vertex[3, tree] + 1]
        @views data_in[:, 2, 2, 1] .= vertices[:, tree_to_vertex[4, tree] + 1]
        @views data_in[:, 1, 1, 2] .= vertices[:, tree_to_vertex[5, tree] + 1]
        @views data_in[:, 2, 1, 2] .= vertices[:, tree_to_vertex[6, tree] + 1]
        @views data_in[:, 1, 2, 2] .= vertices[:, tree_to_vertex[7, tree] + 1]
        @views data_in[:, 2, 2, 2] .= vertices[:, tree_to_vertex[8, tree] + 1]

        # Interpolate corner coordinates to specified nodes
        multiply_dimensionwise!(view(node_coordinates, :, :, :, :, tree),
                                matrix, matrix, matrix,
                                data_in)
    end

    map_node_coordinates!(node_coordinates, mapping)
end

function map_node_coordinates!(node_coordinates::AbstractArray{<:Any, 5}, mapping)
    for tree in axes(node_coordinates, 5),
        k in axes(node_coordinates, 4),
        j in axes(node_coordinates, 3),
        i in axes(node_coordinates, 2)

        node_coordinates[:, i, j, k, tree] .= mapping(node_coordinates[1, i, j, k,
                                                                       tree],
                                                      node_coordinates[2, i, j, k,
                                                                       tree],
                                                      node_coordinates[3, i, j, k,
                                                                       tree])
    end

    return node_coordinates
end

function map_node_coordinates!(node_coordinates::AbstractArray{<:Any, 5},
                               mapping::Nothing)
    return node_coordinates
end

# Calculate physical coordinates of each node of a cubed sphere mesh.
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{<:Any, 5},
                                     nodes, trees_per_face_dimension, layers,
                                     inner_radius, thickness)
    n_cells_x = n_cells_y = trees_per_face_dimension
    n_cells_z = layers

    linear_indices = LinearIndices((n_cells_x, n_cells_y, n_cells_z, 6))

    # Get cell length in reference mesh
    dx = 2 / n_cells_x
    dy = 2 / n_cells_y
    dz = 2 / n_cells_z

    for direction in 1:6
        for cell_z in 1:n_cells_z, cell_y in 1:n_cells_y, cell_x in 1:n_cells_x
            tree = linear_indices[cell_x, cell_y, cell_z, direction]

            x_offset = -1 + (cell_x - 1) * dx + dx / 2
            y_offset = -1 + (cell_y - 1) * dy + dy / 2
            z_offset = -1 + (cell_z - 1) * dz + dz / 2

            for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
                # node_coordinates are the mapped reference node coordinates
                node_coordinates[:, i, j, k, tree] .= cubed_sphere_mapping(x_offset +
                                                                           dx / 2 *
                                                                           nodes[i],
                                                                           y_offset +
                                                                           dy / 2 *
                                                                           nodes[j],
                                                                           z_offset +
                                                                           dz / 2 *
                                                                           nodes[k],
                                                                           inner_radius,
                                                                           thickness,
                                                                           direction)
            end
        end
    end
end

# Map the computational coordinates xi, eta, zeta to the specified side of a cubed sphere
# with the specified inner radius and thickness.
function cubed_sphere_mapping(xi, eta, zeta, inner_radius, thickness, direction)
    alpha = xi * pi / 4
    beta = eta * pi / 4

    # Equiangular projection
    x = tan(alpha)
    y = tan(beta)

    # Coordinates on unit cube per direction, see illustration above in the function connectivity_cubed_sphere
    cube_coordinates = (SVector(-1, -x, y),
                        SVector(1, x, y),
                        SVector(x, -1, y),
                        SVector(-x, 1, y),
                        SVector(-x, y, -1),
                        SVector(x, y, 1))

    # Radius on cube surface
    r = sqrt(1 + x^2 + y^2)

    # Radius of the sphere
    R = inner_radius + thickness * (0.5f0 * (zeta + 1))

    # Projection onto the sphere
    return R / r * cube_coordinates[direction]
end

# Calculate physical coordinates of each element of an unstructured mesh read
# in from a HOHQMesh file. This calculation is done with the transfinite interpolation
# routines found in `mappings_geometry_curved_2d.jl` or `mappings_geometry_straight_2d.jl`
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4},
                                     file_lines::Vector{String}, nodes, vertices, RealT)
    # Get the number of trees and the number of interpolation nodes
    n_trees = last(size(node_coordinates))
    nnodes = length(nodes)

    # Setup the starting file index to read in element indices and the additional
    # curved boundary information provided by HOHQMesh.
    file_idx = findfirst(contains("** mesh polynomial degree"), file_lines) + 1

    # Create a work set of Gamma curves to create the node coordinates
    CurvedSurfaceT = CurvedSurface{RealT}
    surface_curves = Array{CurvedSurfaceT}(undef, 4)

    # Create other work arrays to perform the mesh construction
    element_node_ids = Array{Int}(undef, 4)
    curved_check = Vector{Int}(undef, 4)
    quad_vertices = Array{RealT}(undef, (4, 2))
    quad_vertices_flipped = Array{RealT}(undef, (4, 2))
    curve_values = Array{RealT}(undef, (nnodes, 2))

    # Create the barycentric weights used for the surface interpolations
    bary_weights_ = barycentric_weights(nodes)
    bary_weights = SVector{nnodes}(bary_weights_)

    # Loop through all the trees, i.e., the elements generated by HOHQMesh and create the node coordinates.
    # When we extract information from the `current_line` we start at index 2 in order to
    # avoid the Abaqus comment character "** "
    for tree in 1:n_trees
        # Pull the vertex node IDs
        current_line = split(file_lines[file_idx])
        element_node_ids[1] = parse(Int, current_line[2])
        element_node_ids[2] = parse(Int, current_line[3])
        element_node_ids[3] = parse(Int, current_line[4])
        element_node_ids[4] = parse(Int, current_line[5])

        # Pull the (x,y) values of the four vertices of the current tree out of the global vertices array
        for i in 1:4
            quad_vertices[i, :] .= vertices[1:2, element_node_ids[i]]
        end
        # Pull the information to check if boundary is curved in order to read in additional data
        file_idx += 1
        current_line = split(file_lines[file_idx])
        curved_check[1] = parse(Int, current_line[2])
        curved_check[2] = parse(Int, current_line[3])
        curved_check[3] = parse(Int, current_line[4])
        curved_check[4] = parse(Int, current_line[5])
        if sum(curved_check) == 0
            # Create the node coordinates on this particular element
            calc_node_coordinates!(node_coordinates, tree, nodes, quad_vertices)
        else
            # Quadrilateral element has at least one curved side
            # Flip node ordering to make sure the element is right-handed for the interpolations
            m1 = 1
            m2 = 2
            @views quad_vertices_flipped[1, :] .= quad_vertices[4, :]
            @views quad_vertices_flipped[2, :] .= quad_vertices[2, :]
            @views quad_vertices_flipped[3, :] .= quad_vertices[3, :]
            @views quad_vertices_flipped[4, :] .= quad_vertices[1, :]
            for i in 1:4
                if curved_check[i] == 0
                    # When curved_check[i] is 0 then the "curve" from vertex `i` to vertex `i+1` is a straight line.
                    # Evaluate a linear interpolant between the two points at each of the nodes.
                    for k in 1:nnodes
                        curve_values[k, 1] = linear_interpolate(nodes[k],
                                                                quad_vertices_flipped[m1,
                                                                                      1],
                                                                quad_vertices_flipped[m2,
                                                                                      1])
                        curve_values[k, 2] = linear_interpolate(nodes[k],
                                                                quad_vertices_flipped[m1,
                                                                                      2],
                                                                quad_vertices_flipped[m2,
                                                                                      2])
                    end
                else
                    # When curved_check[i] is 1 this curved boundary information is supplied by the mesh
                    # generator. So we just read it into a work array
                    for k in 1:nnodes
                        file_idx += 1
                        current_line = split(file_lines[file_idx])
                        curve_values[k, 1] = parse(RealT, current_line[2])
                        curve_values[k, 2] = parse(RealT, current_line[3])
                    end
                end
                # Construct the curve interpolant for the current side
                surface_curves[i] = CurvedSurfaceT(nodes, bary_weights,
                                                   copy(curve_values))
                # Indexing update that contains a "flip" to ensure correct element orientation.
                # If we need to construct the straight line "curves" when curved_check[i] == 0
                m1 += 1
                if i == 3
                    m2 = 1
                else
                    m2 += 1
                end
            end
            # Create the node coordinates on this particular element
            calc_node_coordinates!(node_coordinates, tree, nodes, surface_curves)
        end
        # Move file index to the next tree
        file_idx += 1
    end

    return file_idx
end

# Calculate physical coordinates of each element of an unstructured mesh read
# in from a HOHQMesh file. This calculation is done with the transfinite interpolation
# routines found in `transfinite_mappings_3d.jl`
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{<:Any, 5},
                                     file_lines::Vector{String}, nodes, vertices, RealT)
    # Get the number of trees and the number of interpolation nodes
    n_trees = last(size(node_coordinates))
    nnodes = length(nodes)

    # Setup the starting file index to read in element indices and the additional
    # curved boundary information provided by HOHQMesh.
    file_idx = findfirst(contains("** mesh polynomial degree"), file_lines) + 1

    # Create a work set of Gamma curves to create the node coordinates
    CurvedFaceT = CurvedFace{RealT}
    face_curves = Array{CurvedFaceT}(undef, 6)

    # Create other work arrays to perform the mesh construction
    element_node_ids = Array{Int}(undef, 8)
    curved_check = Vector{Int}(undef, 6)
    hex_vertices = Array{RealT}(undef, (3, 8))
    face_vertices = Array{RealT}(undef, (3, 4))
    curve_values = Array{RealT}(undef, (3, nnodes, nnodes))

    # Create the barycentric weights used for the surface interpolations
    bary_weights_ = barycentric_weights(nodes)
    bary_weights = SVector{nnodes}(bary_weights_)

    # Loop through all the trees, i.e., the elements generated by HOHQMesh and create the node coordinates.
    # When we extract information from the `current_line` we start at index 2 in order to
    # avoid the Abaqus comment character "** "
    for tree in 1:n_trees
        # pull the vertex node IDs
        current_line = split(file_lines[file_idx])
        element_node_ids[1] = parse(Int, current_line[2])
        element_node_ids[2] = parse(Int, current_line[3])
        element_node_ids[3] = parse(Int, current_line[4])
        element_node_ids[4] = parse(Int, current_line[5])
        element_node_ids[5] = parse(Int, current_line[6])
        element_node_ids[6] = parse(Int, current_line[7])
        element_node_ids[7] = parse(Int, current_line[8])
        element_node_ids[8] = parse(Int, current_line[9])

        # Pull the (x, y, z) values of the eight vertices of the current tree out of the global vertices array
        for i in 1:8
            hex_vertices[:, i] .= vertices[:, element_node_ids[i]]
        end
        # Pull the information to check if boundary is curved in order to read in additional data
        file_idx += 1
        current_line = split(file_lines[file_idx])
        curved_check[1] = parse(Int, current_line[2])
        curved_check[2] = parse(Int, current_line[3])
        curved_check[3] = parse(Int, current_line[4])
        curved_check[4] = parse(Int, current_line[5])
        curved_check[5] = parse(Int, current_line[6])
        curved_check[6] = parse(Int, current_line[7])
        if sum(curved_check) == 0
            # Create the node coordinates on this element
            calc_node_coordinates!(node_coordinates, tree, nodes, hex_vertices)
        else
            # Hexahedral element has at least one curved side
            for face in 1:6
                if curved_check[face] == 0
                    # Face is a flat plane.
                    # Evaluate a bilinear interpolant between the four vertices
                    # of the face at each of the nodes.
                    get_vertices_for_bilinear_interpolant!(face_vertices, face,
                                                           hex_vertices)
                    for q in 1:nnodes, p in 1:nnodes
                        @views bilinear_interpolation!(curve_values[:, p, q],
                                                       face_vertices, nodes[p],
                                                       nodes[q])
                    end
                else # curved_check[face] == 1
                    # Curved face boundary information is supplied by
                    # the mesh file. Just read it into a work array
                    for q in 1:nnodes, p in 1:nnodes
                        file_idx += 1
                        current_line = split(file_lines[file_idx])
                        curve_values[1, p, q] = parse(RealT, current_line[2])
                        curve_values[2, p, q] = parse(RealT, current_line[3])
                        curve_values[3, p, q] = parse(RealT, current_line[4])
                    end
                end
                # Construct the curve interpolant for the current side
                face_curves[face] = CurvedFaceT(nodes, bary_weights, copy(curve_values))
            end
            # Create the node coordinates on this particular element
            calc_node_coordinates!(node_coordinates, tree, nodes, face_curves)
        end
        # Move file index to the next tree
        file_idx += 1
    end

    return file_idx
end

# Given the eight `hex_vertices` for a hexahedral element extract
# the four `face_vertices` for a particular `face_index`.
function get_vertices_for_bilinear_interpolant!(face_vertices, face_index, hex_vertices)
    if face_index == 1
        @views face_vertices[:, 1] .= hex_vertices[:, 1]
        @views face_vertices[:, 2] .= hex_vertices[:, 2]
        @views face_vertices[:, 3] .= hex_vertices[:, 6]
        @views face_vertices[:, 4] .= hex_vertices[:, 5]
    elseif face_index == 2
        @views face_vertices[:, 1] .= hex_vertices[:, 4]
        @views face_vertices[:, 2] .= hex_vertices[:, 3]
        @views face_vertices[:, 3] .= hex_vertices[:, 7]
        @views face_vertices[:, 4] .= hex_vertices[:, 8]
    elseif face_index == 3
        @views face_vertices[:, 1] .= hex_vertices[:, 1]
        @views face_vertices[:, 2] .= hex_vertices[:, 2]
        @views face_vertices[:, 3] .= hex_vertices[:, 3]
        @views face_vertices[:, 4] .= hex_vertices[:, 4]
    elseif face_index == 4
        @views face_vertices[:, 1] .= hex_vertices[:, 2]
        @views face_vertices[:, 2] .= hex_vertices[:, 3]
        @views face_vertices[:, 3] .= hex_vertices[:, 6]
        @views face_vertices[:, 4] .= hex_vertices[:, 7]
    elseif face_index == 5
        @views face_vertices[:, 1] .= hex_vertices[:, 5]
        @views face_vertices[:, 2] .= hex_vertices[:, 6]
        @views face_vertices[:, 3] .= hex_vertices[:, 7]
        @views face_vertices[:, 4] .= hex_vertices[:, 8]
    else # face_index == 6
        @views face_vertices[:, 1] .= hex_vertices[:, 1]
        @views face_vertices[:, 2] .= hex_vertices[:, 4]
        @views face_vertices[:, 3] .= hex_vertices[:, 8]
        @views face_vertices[:, 4] .= hex_vertices[:, 5]
    end
end

# Evaluate a bilinear interpolant at a point (u,v) given the four vertices where the face is right-handed
#      4                3
#      o----------------o
#      |                |
#      |                |
#      |                |
#      |                |
#      |                |
#      |                |
#      o----------------o
#      1                2
# and return the 3D coordinate point (x, y, z)
function bilinear_interpolation!(coordinate, face_vertices, u, v)
    for j in 1:3
        coordinate[j] = 0.25f0 * (face_vertices[j, 1] * (1 - u) * (1 - v)
                         + face_vertices[j, 2] * (1 + u) * (1 - v)
                         + face_vertices[j, 3] * (1 + u) * (1 + v)
                         + face_vertices[j, 4] * (1 - u) * (1 + v))
    end
end

function get_global_first_element_ids(mesh::P4estMesh)
    return unsafe_wrap(Array, mesh.p4est.global_first_quadrant, mpi_nranks() + 1)
end

function balance!(mesh::P4estMesh{2}, init_fn = C_NULL)
    p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, init_fn)
    # Due to a bug in `p4est`, the forest needs to be rebalanced twice sometimes
    # See https://github.com/cburstedde/p4est/issues/112
    p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, init_fn)
end

function balance!(mesh::P4estMesh{3}, init_fn = C_NULL)
    p8est_balance(mesh.p4est, P8EST_CONNECT_FACE, init_fn)
end

function partition!(mesh::P4estMesh{2}; weight_fn = C_NULL)
    p4est_partition(mesh.p4est, Int(mesh.p4est_partition_allow_for_coarsening),
                    weight_fn)
end

function partition!(mesh::P4estMesh{3}; weight_fn = C_NULL)
    p8est_partition(mesh.p4est, Int(mesh.p4est_partition_allow_for_coarsening),
                    weight_fn)
end

function update_ghost_layer!(mesh::P4estMesh)
    ghost_destroy_p4est(mesh.ghost)
    mesh.ghost = PointerWrapper(ghost_new_p4est(mesh.p4est))
end

function init_fn(p4est, which_tree, quadrant)
    # Unpack quadrant's user data ([global quad ID, controller_value])
    # Use `unsafe_load` here since `quadrant.p.user_data isa Ptr{Ptr{Nothing}}`
    # and we only need the first (only!) entry
    pw = PointerWrapper(Int, unsafe_load(quadrant.p.user_data))

    # Initialize quad ID as -1 and controller_value as 0 (don't refine or coarsen)
    pw[1] = -1
    pw[2] = 0
    return nothing
end

# 2D
function cfunction(::typeof(init_fn), ::Val{2})
    @cfunction(init_fn, Cvoid,
               (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{p4est_quadrant_t}))
end
# 3D
function cfunction(::typeof(init_fn), ::Val{3})
    @cfunction(init_fn, Cvoid,
               (Ptr{p8est_t}, Ptr{p4est_topidx_t}, Ptr{p8est_quadrant_t}))
end

function refine_fn(p4est, which_tree, quadrant)
    # Controller value has been copied to the quadrant's user data storage before.
    # Unpack quadrant's user data ([global quad ID, controller_value]).
    # Use `unsafe_load` here since `quadrant.p.user_data isa Ptr{Ptr{Nothing}}`
    # and we only need the first (only!) entry
    pw = PointerWrapper(Int, unsafe_load(quadrant.p.user_data))
    controller_value = pw[2]

    if controller_value > 0
        # return true (refine)
        return Cint(1)
    else
        # return false (don't refine)
        return Cint(0)
    end
end

# 2D
function cfunction(::typeof(refine_fn), ::Val{2})
    @cfunction(refine_fn, Cint,
               (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{p4est_quadrant_t}))
end
# 3D
function cfunction(::typeof(refine_fn), ::Val{3})
    @cfunction(refine_fn, Cint,
               (Ptr{p8est_t}, Ptr{p4est_topidx_t}, Ptr{p8est_quadrant_t}))
end

# Refine marked cells and rebalance forest.
# Return a list of all cells that have been refined during refinement or rebalancing.
function refine!(mesh::P4estMesh)
    # Copy original element IDs to quad user data storage
    original_n_cells = ncells(mesh)
    save_original_ids(mesh)

    init_fn_c = cfunction(init_fn, Val(ndims(mesh)))
    refine_fn_c = cfunction(refine_fn, Val(ndims(mesh)))

    # Refine marked cells
    @trixi_timeit timer() "refine" refine_p4est!(mesh.p4est, false, refine_fn_c,
                                                 init_fn_c)

    @trixi_timeit timer() "rebalance" balance!(mesh, init_fn_c)

    return collect_changed_cells(mesh, original_n_cells)
end

function coarsen_fn(p4est, which_tree, quadrants_ptr)
    quadrants = unsafe_wrap_quadrants(quadrants_ptr, p4est)

    # Controller value has been copied to the quadrant's user data storage before.
    # Load controller value from quadrant's user data ([global quad ID, controller_value]).
    # Use `unsafe_load` here since `quadrant.p.user_data isa Ptr{Ptr{Nothing}}`
    # and we only need the first (only!) entry
    controller_value(i) = PointerWrapper(Int, unsafe_load(quadrants[i].p.user_data))[2]

    # `p4est` calls this function for each 2^ndims quads that could be coarsened to a single one.
    # Only coarsen if all these 2^ndims quads have been marked for coarsening.
    if all(i -> controller_value(i) < 0, eachindex(quadrants))
        # return true (coarsen)
        return Cint(1)
    else
        # return false (don't coarsen)
        return Cint(0)
    end
end

# 2D
function unsafe_wrap_quadrants(quadrants_ptr, ::Ptr{p4est_t})
    unsafe_wrap(Array, quadrants_ptr, 4)
end
# 3D
function unsafe_wrap_quadrants(quadrants_ptr, ::Ptr{p8est_t})
    unsafe_wrap(Array, quadrants_ptr, 8)
end

# 2D
function cfunction(::typeof(coarsen_fn), ::Val{2})
    @cfunction(coarsen_fn, Cint,
               (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{Ptr{p4est_quadrant_t}}))
end
# 3D
function cfunction(::typeof(coarsen_fn), ::Val{3})
    @cfunction(coarsen_fn, Cint,
               (Ptr{p8est_t}, Ptr{p4est_topidx_t}, Ptr{Ptr{p8est_quadrant_t}}))
end

# Coarsen marked cells if the forest will stay balanced.
# Return a list of all cells that have been coarsened.
function coarsen!(mesh::P4estMesh)
    # Copy original element IDs to quad user data storage
    original_n_cells = ncells(mesh)
    save_original_ids(mesh)

    # Coarsen marked cells
    coarsen_fn_c = cfunction(coarsen_fn, Val(ndims(mesh)))
    init_fn_c = cfunction(init_fn, Val(ndims(mesh)))

    @trixi_timeit timer() "coarsen!" coarsen_p4est!(mesh.p4est, false, coarsen_fn_c,
                                                    init_fn_c)

    # IDs of newly created cells (one-based)
    new_cells = collect_new_cells(mesh)
    # Old IDs of cells that have been coarsened (one-based)
    coarsened_cells_vec = collect_changed_cells(mesh, original_n_cells)
    # 2^ndims changed cells should have been coarsened to one new cell.
    # This matrix will store the IDs of all cells that have been coarsened to cell new_cells[i]
    # in the i-th column.
    coarsened_cells = reshape(coarsened_cells_vec, 2^ndims(mesh), length(new_cells))

    # Save new original IDs to find out what changed after balancing
    intermediate_n_cells = ncells(mesh)
    save_original_ids(mesh)

    @trixi_timeit timer() "rebalance" balance!(mesh, init_fn_c)

    refined_cells = collect_changed_cells(mesh, intermediate_n_cells)

    # Some cells may have been coarsened even though they unbalanced the forest.
    # These cells have now been refined again by p4est_balance.
    # refined_cells contains the intermediate IDs (ID of coarse cell
    # between coarsening and balancing) of these cells.
    # Find original ID of each cell that has been coarsened and then refined again.
    for refined_cell in refined_cells
        # i-th cell of the ones that have been created by coarsening has been refined again
        i = findfirst(==(refined_cell), new_cells)

        # Remove IDs of the 2^ndims cells that have been coarsened to this cell
        coarsened_cells[:, i] .= -1
    end

    # Return all IDs of cells that have been coarsened but not refined again by balancing
    return coarsened_cells_vec[coarsened_cells_vec .>= 0]
end

# Copy global quad ID to quad's user data storage, will be called below
function save_original_id_iter_volume(info, user_data)
    info_pw = PointerWrapper(info)

    # Load tree from global trees array, one-based indexing
    tree_pw = load_pointerwrapper_tree(info_pw.p4est, info_pw.treeid[] + 1)
    # Quadrant numbering offset of this quadrant
    offset = tree_pw.quadrants_offset[]
    # Global quad ID
    quad_id = offset + info_pw.quadid[]

    # Unpack quadrant's user data ([global quad ID, controller_value])
    pw = PointerWrapper(Int, info_pw.quad.p.user_data[])
    # Save global quad ID
    pw[1] = quad_id
    return nothing
end

# 2D
function cfunction(::typeof(save_original_id_iter_volume), ::Val{2})
    @cfunction(save_original_id_iter_volume, Cvoid,
               (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(save_original_id_iter_volume), ::Val{3})
    @cfunction(save_original_id_iter_volume, Cvoid,
               (Ptr{p8est_iter_volume_info_t}, Ptr{Cvoid}))
end

# Copy old element IDs to each quad's user data storage
function save_original_ids(mesh::P4estMesh)
    iter_volume_c = cfunction(save_original_id_iter_volume, Val(ndims(mesh)))

    iterate_p4est(mesh.p4est, C_NULL; iter_volume_c = iter_volume_c)
end

# Extract information about which cells have been changed
function collect_changed_iter_volume(info, user_data)
    info_pw = PointerWrapper(info)

    # The original element ID has been saved to user_data before.
    # Load original quad ID from quad's user data ([global quad ID, controller_value]).
    quad_data_pw = PointerWrapper(Int, info_pw.quad.p.user_data[])
    original_id = quad_data_pw[1]

    # original_id of cells that have been newly created is -1
    if original_id >= 0
        # Unpack user_data = original_cells
        user_data_pw = PointerWrapper(Int, user_data)

        # If quad has an original_id, it existed before refinement/coarsening,
        # and therefore wasn't changed.
        # Mark original_id as "not changed during refinement/coarsening" in original_cells
        user_data_pw[original_id + 1] = 0
    end
    return nothing
end

# 2D
function cfunction(::typeof(collect_changed_iter_volume), ::Val{2})
    @cfunction(collect_changed_iter_volume, Cvoid,
               (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(collect_changed_iter_volume), ::Val{3})
    @cfunction(collect_changed_iter_volume, Cvoid,
               (Ptr{p8est_iter_volume_info_t}, Ptr{Cvoid}))
end

function collect_changed_cells(mesh::P4estMesh, original_n_cells)
    original_cells = collect(1:original_n_cells)

    # Iterate over all quads and set original cells that haven't been changed to zero
    iter_volume_c = cfunction(collect_changed_iter_volume, Val(ndims(mesh)))

    iterate_p4est(mesh.p4est, original_cells; iter_volume_c = iter_volume_c)

    # Changed cells are all that haven't been set to zero above
    changed_original_cells = original_cells[original_cells .> 0]

    return changed_original_cells
end

# Extract newly created cells
function collect_new_iter_volume(info, user_data)
    info_pw = PointerWrapper(info)

    # The original element ID has been saved to user_data before.
    # Unpack quadrant's user data ([global quad ID, controller_value]).
    original_id = PointerWrapper(Int, info_pw.quad.p.user_data[])[1]

    # original_id of cells that have been newly created is -1
    if original_id < 0
        # Load tree from global trees array, one-based indexing
        tree_pw = load_pointerwrapper_tree(info_pw.p4est, info_pw.treeid[] + 1)
        # Quadrant numbering offset of this quadrant
        offset = tree_pw.quadrants_offset[]
        # Global quad ID
        quad_id = offset + info_pw.quadid[]

        # Unpack user_data = original_cells
        user_data_pw = PointerWrapper(Int, user_data)

        # Mark cell as "newly created during refinement/coarsening/balancing"
        user_data_pw[quad_id + 1] = 1
    end
    return nothing
end

# 2D
function cfunction(::typeof(collect_new_iter_volume), ::Val{2})
    @cfunction(collect_new_iter_volume, Cvoid,
               (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(collect_new_iter_volume), ::Val{3})
    @cfunction(collect_new_iter_volume, Cvoid,
               (Ptr{p8est_iter_volume_info_t}, Ptr{Cvoid}))
end

function collect_new_cells(mesh::P4estMesh)
    cell_is_new = zeros(Int, ncells(mesh))

    # Iterate over all quads and set original cells that have been changed to one
    iter_volume_c = cfunction(collect_new_iter_volume, Val(ndims(mesh)))

    iterate_p4est(mesh.p4est, cell_is_new; iter_volume_c = iter_volume_c)

    # Changed cells are all that haven't been set to zero above
    new_cells = findall(==(1), cell_is_new)

    return new_cells
end
end # @muladd
