"""
    T8codeMesh{NDIMS} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees that uses the C library 
['t8code'](https://github.com/DLR-AMR/t8code)
to manage trees and mesh refinement.
"""
mutable struct T8codeMesh{NDIMS, RealT <: Real, IsParallel, NDIMSP2, NNODES} <:
               AbstractMesh{NDIMS}
    cmesh       :: Ptr{t8_cmesh} # cpointer to coarse mesh
    scheme      :: Ptr{t8_eclass_scheme} # cpointer to element scheme
    forest      :: Ptr{t8_forest} # cpointer to forest
    is_parallel :: IsParallel

    # This specifies the geometry interpolation for each tree.
    tree_node_coordinates::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree]

    # Stores the quadrature nodes.
    nodes::SVector{NNODES, RealT}

    boundary_names   :: Array{Symbol, 2}      # [face direction, tree]
    current_filename :: String

    ninterfaces :: Int
    nmortars    :: Int
    nboundaries :: Int

    function T8codeMesh{NDIMS}(cmesh, scheme, forest, tree_node_coordinates, nodes,
                               boundary_names,
                               current_filename) where {NDIMS}
        is_parallel = False()

        mesh = new{NDIMS, Float64, typeof(is_parallel), NDIMS + 2, length(nodes)}(cmesh,
                                                                                  scheme,
                                                                                  forest,
                                                                                  is_parallel)

        mesh.nodes = nodes
        mesh.boundary_names = boundary_names
        mesh.current_filename = current_filename
        mesh.tree_node_coordinates = tree_node_coordinates

        finalizer(mesh) do mesh
            # When finalizing `mesh.forest`, `mesh.scheme` and `mesh.cmesh` are
            # also cleaned up from within `t8code`. The cleanup code for
            # `cmesh` does some MPI calls for deallocating shared memory
            # arrays. Due to garbage collection in Julia the order of shutdown
            # is not deterministic. The following code might happen after MPI
            # is already in finalized state.
            # If the environment variable `TRIXI_T8CODE_SC_FINALIZE` is set the
            # `finalize_hook` of the MPI module takes care of the cleanup. See
            # further down. However, this might cause a pile-up of `mesh`
            # objects during long-running sessions.
            if !MPI.Finalized()
                trixi_t8_unref_forest(mesh.forest)
            end
        end

        # This finalizer call is only recommended during development and not for
        # production runs, especially long-running sessions since a reference to
        # the `mesh` object will be kept throughout the lifetime of the session.
        # See comments in `init_t8code()` in file `src/auxiliary/t8code.jl` for
        # more information.
        if haskey(ENV, "TRIXI_T8CODE_SC_FINALIZE")
            MPI.add_finalize_hook!() do
                trixi_t8_unref_forest(mesh.forest)
            end
        end

        return mesh
    end
end

const SerialT8codeMesh{NDIMS} = T8codeMesh{NDIMS, <:Real, <:False}
@inline mpi_parallel(mesh::SerialT8codeMesh) = False()

@inline Base.ndims(::T8codeMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::T8codeMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline ntrees(mesh::T8codeMesh) = Int(t8_forest_get_num_local_trees(mesh.forest))
@inline ncells(mesh::T8codeMesh) = Int(t8_forest_get_local_num_elements(mesh.forest))
@inline ninterfaces(mesh::T8codeMesh) = mesh.ninterfaces
@inline nmortars(mesh::T8codeMesh) = mesh.nmortars
@inline nboundaries(mesh::T8codeMesh) = mesh.nboundaries

function Base.show(io::IO, mesh::T8codeMesh)
    print(io, "T8codeMesh{", ndims(mesh), ", ", real(mesh), "}")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::T8codeMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        setup = [
            "#trees" => ntrees(mesh),
            "current #cells" => ncells(mesh),
            "polydeg" => length(mesh.nodes) - 1,
        ]
        summary_box(io,
                    "T8codeMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) * "}",
                    setup)
    end
end

"""
    T8codeMesh(trees_per_dimension; polydeg, mapping=identity,
               RealT=Float64, initial_refinement_level=0, periodicity=true)

Create a structured potentially curved 'T8codeMesh' of the specified size.

Non-periodic boundaries will be called ':x_neg', ':x_pos', ':y_neg', ':y_pos', ':z_neg', ':z_pos'.

# Arguments
- 'trees_per_dimension::NTupleE{NDIMS, Int}': the number of trees in each dimension.
- 'polydeg::Integer': polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
- 'mapping': a function of 'NDIMS' variables to describe the mapping that transforms
             the reference mesh ('[-1, 1]^n') to the physical domain.
- 'RealT::Type': the type that should be used for coordinates.
- 'initial_refinement_level::Integer': refine the mesh uniformly to this level before the simulation starts.
- 'periodicity': either a 'Bool' deciding if all of the boundaries are periodic or an 'NTuple{NDIMS, Bool}'
                 deciding for each dimension if the boundaries in this dimension are periodic.
"""
function T8codeMesh(trees_per_dimension; polydeg,
                    mapping = coordinates2mapping((-1.0, -1.0), (1.0, 1.0)),
                    RealT = Float64, initial_refinement_level = 0, periodicity = true)
    NDIMS = length(trees_per_dimension)

    @assert NDIMS == 2 # Only support for NDIMS = 2 yet.

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

    conn = T8code.Libt8.p4est_connectivity_new_brick(trees_per_dimension..., periodicity...)
    do_partition = 0
    cmesh = t8_cmesh_new_from_p4est(conn, mpi_comm(), do_partition)
    T8code.Libt8.p4est_connectivity_destroy(conn)

    scheme = t8_scheme_new_default_cxx()
    forest = t8_forest_new_uniform(cmesh, scheme, initial_refinement_level, 0, mpi_comm())

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes

    tree_node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> length(nodes), NDIMS)...,
                                                    prod(trees_per_dimension))

    # Get cell length in reference mesh: Omega_ref = [-1,1]^2.
    dx = 2 / trees_per_dimension[1]
    dy = 2 / trees_per_dimension[2]

    num_local_trees = t8_cmesh_get_num_local_trees(cmesh)

    # Non-periodic boundaries.
    boundary_names = fill(Symbol("---"), 2 * NDIMS, prod(trees_per_dimension))

    for itree in 1:num_local_trees
        veptr = t8_cmesh_get_tree_vertices(cmesh, itree - 1)
        verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))

        # Calculate node coordinates of reference mesh.
        cell_x_offset = (verts[1, 1] - 1 / 2 * (trees_per_dimension[1] - 1)) * dx
        cell_y_offset = (verts[2, 1] - 1 / 2 * (trees_per_dimension[2] - 1)) * dy

        for j in eachindex(nodes), i in eachindex(nodes)
            tree_node_coordinates[:, i, j, itree] .= mapping(cell_x_offset +
                                                             dx * nodes[i] / 2,
                                                             cell_y_offset +
                                                             dy * nodes[j] / 2)
        end

        if !periodicity[1]
            boundary_names[1, itree] = :x_neg
            boundary_names[2, itree] = :x_pos
        end

        if !periodicity[2]
            boundary_names[3, itree] = :y_neg
            boundary_names[4, itree] = :y_pos
        end
    end

    return T8codeMesh{NDIMS}(cmesh, scheme, forest, tree_node_coordinates, nodes,
                             boundary_names, "")
end

"""
    T8codeMesh{NDIMS}(cmesh::Ptr{t8_cmesh},
                     mapping=nothing, polydeg=1, RealT=Float64,
                     initial_refinement_level=0)

Main mesh constructor for the `T8codeMesh` that imports an unstructured,
conforming mesh from a `t8_cmesh` data structure.

# Arguments
- `cmesh::Ptr{t8_cmesh}`: Pointer to a cmesh object.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
"""
function T8codeMesh{NDIMS}(cmesh::Ptr{t8_cmesh};
                           mapping = nothing, polydeg = 1, RealT = Float64,
                           initial_refinement_level = 0) where {NDIMS}
    @assert NDIMS == 2 # Only support for NDIMS = 2 yet.

    scheme = t8_scheme_new_default_cxx()
    forest = t8_forest_new_uniform(cmesh, scheme, initial_refinement_level, 0, mpi_comm())

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes

    num_local_trees = t8_cmesh_get_num_local_trees(cmesh)

    tree_node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> length(nodes), NDIMS)...,
                                                    num_local_trees)

    nodes_in = [-1.0, 1.0]
    matrix = polynomial_interpolation_matrix(nodes_in, nodes)
    data_in = Array{RealT, 3}(undef, 2, 2, 2)
    tmp1 = zeros(RealT, 2, length(nodes), length(nodes_in))

    for itree in 0:(num_local_trees - 1)
        veptr = t8_cmesh_get_tree_vertices(cmesh, itree)
        verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))

        u = verts[:, 2] - verts[:, 1]
        v = verts[:, 3] - verts[:, 1]
        w = [0.0, 0.0, 1.0]

        vol = dot(cross(u, v), w)

        if vol < 0.0
            @warn "Discovered negative volumes in `cmesh`: vol = $vol"
        end

        # Tree vertices are stored in z-order.
        @views data_in[:, 1, 1] .= verts[1:2, 1]
        @views data_in[:, 2, 1] .= verts[1:2, 2]
        @views data_in[:, 1, 2] .= verts[1:2, 3]
        @views data_in[:, 2, 2] .= verts[1:2, 4]

        # Interpolate corner coordinates to specified nodes.
        multiply_dimensionwise!(view(tree_node_coordinates, :, :, :, itree + 1),
                                matrix, matrix,
                                data_in,
                                tmp1)
    end

    map_node_coordinates!(tree_node_coordinates, mapping)

    # There's no simple and generic way to distinguish boundaries. Name all of them :all.
    boundary_names = fill(:all, 2 * NDIMS, num_local_trees)

    return T8codeMesh{NDIMS}(cmesh, scheme, forest, tree_node_coordinates, nodes,
                             boundary_names, "")
end

"""
    T8codeMesh{NDIMS}(conn::Ptr{p4est_connectivity},
                      mapping=nothing, polydeg=1, RealT=Float64,
                      initial_refinement_level=0)

Main mesh constructor for the `T8codeMesh` that imports an unstructured,
conforming mesh from a `p4est_connectivity` data structure.

# Arguments
- `conn::Ptr{p4est_connectivity}`: Pointer to a P4est connectivity object.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
"""
function T8codeMesh{NDIMS}(conn::Ptr{p4est_connectivity}; kwargs...) where {NDIMS}
    @assert NDIMS == 2 # Only support for NDIMS = 2 yet.

    cmesh = t8_cmesh_new_from_p4est(conn, mpi_comm(), 0)

    return T8codeMesh{NDIMS}(cmesh; kwargs...)
end

"""
    T8codeMesh{NDIMS}(meshfile::String;
                     mapping=nothing, polydeg=1, RealT=Float64,
                     initial_refinement_level=0)

Main mesh constructor for the `T8codeMesh` that imports an unstructured, conforming
mesh from a Gmsh mesh file (`.msh`).

# Arguments
- `meshfile::String`: path to a Gmsh mesh file.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
"""
function T8codeMesh{NDIMS}(meshfile::String; kwargs...) where {NDIMS}
    @assert NDIMS == 2 # Only support for NDIMS = 2 yet.

    # Prevent `t8code` from crashing Julia if the file doesn't exist.
    @assert isfile(meshfile)

    meshfile_prefix, meshfile_suffix = splitext(meshfile)

    cmesh = t8_cmesh_from_msh_file(meshfile_prefix, 0, mpi_comm(), NDIMS, 0, 0)

    return T8codeMesh{NDIMS}(cmesh; kwargs...)
end

# TODO: Just a placeholder. Will be implemented later when MPI is supported.
function balance!(mesh::T8codeMesh, init_fn = C_NULL)
    return nothing
end

# TODO: Just a placeholder. Will be implemented later when MPI is supported.
function partition!(mesh::T8codeMesh; allow_coarsening = true, weight_fn = C_NULL)
    return nothing
end
