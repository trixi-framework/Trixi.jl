"""
    T8codeMesh{NDIMS} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees that uses the C library 
['t8code'](https://github.com/DLR-AMR/t8code)
to manage trees and mesh refinement.
"""
mutable struct T8codeMesh{NDIMS, RealT <: Real, IsParallel, NDIMSP2, NNODES} <:
               AbstractMesh{NDIMS}
    forest      :: Ptr{t8_forest} # cpointer to forest
    is_parallel :: IsParallel

    unsaved_changes::Bool

    # This specifies the geometry interpolation for each tree.
    tree_node_coordinates::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree]

    # Stores the quadrature nodes.
    nodes::SVector{NNODES, RealT}

    boundary_names   :: Array{Symbol, 2}      # [face direction, tree]
    current_filename :: String

    ninterfaces :: Int
    nmortars    :: Int
    nboundaries :: Int

    nmpiinterfaces :: Int
    nmpimortars    :: Int

    function T8codeMesh{NDIMS}(forest, tree_node_coordinates, nodes,
                               boundary_names,
                               current_filename) where {NDIMS}
        is_parallel = mpi_isparallel() ? True() : False()

        mesh = new{NDIMS, Float64, typeof(is_parallel), NDIMS + 2, length(nodes)}(forest,
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
const ParallelT8codeMesh{NDIMS} = T8codeMesh{NDIMS, <:Real, <:True}
@inline mpi_parallel(mesh::SerialT8codeMesh) = False()
@inline mpi_parallel(mesh::ParallelT8codeMesh) = True()

@inline Base.ndims(::T8codeMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::T8codeMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline ntrees(mesh::T8codeMesh) = size(mesh.tree_node_coordinates)[end]
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
- 'RealT::Type': the type that should be used for coordinates.
- 'initial_refinement_level::Integer': refine the mesh uniformly to this level before the simulation starts.
- 'periodicity': either a 'Bool' deciding if all of the boundaries are periodic or an 'NTuple{NDIMS, Bool}'
                 deciding for each dimension if the boundaries in this dimension are periodic.
"""
function T8codeMesh(trees_per_dimension; polydeg,
                    mapping = nothing, faces = nothing, coordinates_min = nothing,
                    coordinates_max = nothing,
                    RealT = Float64, initial_refinement_level = 0,
                    periodicity = true)
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

    if NDIMS == 2
        conn = T8code.Libt8.p4est_connectivity_new_brick(trees_per_dimension...,
                                                         periodicity...)
        do_partition = 0
        cmesh = t8_cmesh_new_from_p4est(conn, mpi_comm(), do_partition)
        T8code.Libt8.p4est_connectivity_destroy(conn)
    elseif NDIMS == 3
        conn = T8code.Libt8.p8est_connectivity_new_brick(trees_per_dimension...,
                                                         periodicity...)
        do_partition = 0
        cmesh = t8_cmesh_new_from_p8est(conn, mpi_comm(), do_partition)
        T8code.Libt8.p8est_connectivity_destroy(conn)
    end

    do_face_ghost = mpi_isparallel()
    scheme = t8_scheme_new_default_cxx()
    forest = t8_forest_new_uniform(cmesh, scheme, initial_refinement_level, do_face_ghost,
                                   mpi_comm())

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes

    num_trees = t8_cmesh_get_num_trees(cmesh)

    tree_node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> length(nodes), NDIMS)...,
                                                    num_trees)

    # Get cell length in reference mesh: Omega_ref = [-1,1]^2.
    dx = [2 / n for n in trees_per_dimension]

    # Non-periodic boundaries.
    boundary_names = fill(Symbol("---"), 2 * NDIMS, prod(trees_per_dimension))

    if mapping === nothing
        mapping_ = coordinates2mapping(ntuple(_ -> -1.0, NDIMS), ntuple(_ -> 1.0, NDIMS))
    else
        mapping_ = mapping
    end

    for itree in 1:num_trees
        veptr = t8_cmesh_get_tree_vertices(cmesh, itree - 1)
        verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))

        if NDIMS == 2
            # Calculate node coordinates of reference mesh for 2D.
            cell_x_offset = (verts[1, 1] - 0.5 * (trees_per_dimension[1] - 1)) * dx[1]
            cell_y_offset = (verts[2, 1] - 0.5 * (trees_per_dimension[2] - 1)) * dx[2]

            for j in eachindex(nodes), i in eachindex(nodes)
                tree_node_coordinates[:, i, j, itree] .= mapping_(cell_x_offset +
                                                                  dx[1] * nodes[i] / 2,
                                                                  cell_y_offset +
                                                                  dx[2] * nodes[j] / 2)
            end
        elseif NDIMS == 3
            # Calculate node coordinates of reference mesh for 2D.
            cell_x_offset = (verts[1, 1] - 0.5 * (trees_per_dimension[1] - 1)) * dx[1]
            cell_y_offset = (verts[2, 1] - 0.5 * (trees_per_dimension[2] - 1)) * dx[2]
            cell_z_offset = (verts[3, 1] - 0.5 * (trees_per_dimension[3] - 1)) * dx[3]

            for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
                tree_node_coordinates[:, i, j, k, itree] .= mapping_(cell_x_offset +
                                                                     dx[1] * nodes[i] / 2,
                                                                     cell_y_offset +
                                                                     dx[2] * nodes[j] / 2,
                                                                     cell_z_offset +
                                                                     dx[3] * nodes[k] / 2)
            end
        else
            throw(ArgumentError("NDIMS should be 2 or 3."))
        end

        if !periodicity[1]
            boundary_names[1, itree] = :x_neg
            boundary_names[2, itree] = :x_pos
        end

        if !periodicity[2]
            boundary_names[3, itree] = :y_neg
            boundary_names[4, itree] = :y_pos
        end

        if NDIMS > 2
            if !periodicity[3]
                boundary_names[5, itree] = :z_neg
                boundary_names[6, itree] = :z_pos
            end
        end
    end

    return T8codeMesh{NDIMS}(forest, tree_node_coordinates, nodes,
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
    do_face_ghost = mpi_isparallel()
    scheme = t8_scheme_new_default_cxx()
    forest = t8_forest_new_uniform(cmesh, scheme, initial_refinement_level, do_face_ghost,
                                   mpi_comm())

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes

    num_trees = t8_cmesh_get_num_trees(cmesh)

    tree_node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> length(nodes), NDIMS)...,
                                                    num_trees)

    nodes_in = [-1.0, 1.0]
    matrix = polynomial_interpolation_matrix(nodes_in, nodes)

    if NDIMS == 2
        data_in = Array{RealT, 3}(undef, 2, 2, 2)
        tmp1 = zeros(RealT, 2, length(nodes), length(nodes_in))

        for itree in 0:(num_trees - 1)
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

    elseif NDIMS == 3
        data_in = Array{RealT, 4}(undef, 3, 2, 2, 2)
        tmp1 = zeros(RealT, 3, length(nodes), length(nodes_in), length(nodes_in))

        for itree in 0:(num_trees - 1)
            veptr = t8_cmesh_get_tree_vertices(cmesh, itree)
            verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))

            # Tree vertices are stored in z-order.
            @views data_in[:, 1, 1, 1] .= verts[1:3, 1]
            @views data_in[:, 2, 1, 1] .= verts[1:3, 2]
            @views data_in[:, 1, 2, 1] .= verts[1:3, 3]
            @views data_in[:, 2, 2, 1] .= verts[1:3, 4]

            @views data_in[:, 1, 1, 2] .= verts[1:3, 5]
            @views data_in[:, 2, 1, 2] .= verts[1:3, 6]
            @views data_in[:, 1, 2, 2] .= verts[1:3, 7]
            @views data_in[:, 2, 2, 2] .= verts[1:3, 8]

            # Interpolate corner coordinates to specified nodes.
            multiply_dimensionwise!(view(tree_node_coordinates, :, :, :, :, itree + 1),
                                    matrix, matrix, matrix,
                                    data_in,
                                    tmp1)
        end
    else
        throw(ArgumentError("NDIMS should be 2 or 3."))
    end

    map_node_coordinates!(tree_node_coordinates, mapping)

    # There's no simple and generic way to distinguish boundaries. Name all of them :all.
    boundary_names = fill(:all, 2 * NDIMS, num_trees)

    return T8codeMesh{NDIMS}(forest, tree_node_coordinates, nodes,
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
function T8codeMesh(conn::Ptr{p4est_connectivity}; kwargs...)
    cmesh = t8_cmesh_new_from_p4est(conn, mpi_comm(), 0)

    return T8codeMesh{2}(cmesh; kwargs...)
end

"""
    T8codeMesh{NDIMS}(conn::Ptr{p8est_connectivity},
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
function T8codeMesh(conn::Ptr{p8est_connectivity}; kwargs...)
    cmesh = t8_cmesh_new_from_p8est(conn, mpi_comm(), 0)

    return T8codeMesh{3}(cmesh; kwargs...)
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
    # Prevent `t8code` from crashing Julia if the file doesn't exist.
    @assert isfile(meshfile)

    meshfile_prefix, meshfile_suffix = splitext(meshfile)

    cmesh = t8_cmesh_from_msh_file(meshfile_prefix, 0, mpi_comm(), NDIMS, 0, 0)

    return T8codeMesh{NDIMS}(cmesh; kwargs...)
end

# Compute the global ids (zero-indexed) of first element in each MPI rank.
function get_global_first_element_ids(mesh::T8codeMesh)
    n_elements_local = Int(t8_forest_get_local_num_elements(mesh.forest))
    n_elements_by_rank = Vector{Int}(undef, mpi_nranks())
    n_elements_by_rank[mpi_rank() + 1] = n_elements_local
    MPI.Allgather!(MPI.UBuffer(n_elements_by_rank, 1), mpi_comm())
    return [sum(n_elements_by_rank[1:(rank - 1)]) for rank in 1:mpi_nranks()+1]
end

function balance!(mesh::T8codeMesh)
    new_forest_ref = Ref{t8_forest_t}()
    t8_forest_init(new_forest_ref)
    new_forest = new_forest_ref[]

    let set_from = mesh.forest, no_repartition = 1, do_ghost = 1
        t8_forest_set_balance(new_forest, set_from, no_repartition)
        t8_forest_set_ghost(new_forest, do_ghost, T8_GHOST_FACES)
        t8_forest_commit(new_forest)
    end

    mesh.forest = new_forest

    return nothing
end

function partition!(mesh::T8codeMesh; allow_coarsening = true, weight_fn = C_NULL)
    new_forest_ref = Ref{t8_forest_t}()
    t8_forest_init(new_forest_ref)
    new_forest = new_forest_ref[]

    let set_from = mesh.forest, set_for_coarsening = 1, do_ghost = 1
        t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
        t8_forest_set_ghost(new_forest, do_ghost, T8_GHOST_FACES)
        t8_forest_commit(new_forest)
    end

    mesh.forest = new_forest

    return nothing
end
