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

    # This specifies the geometry interpolation for each tree.
    tree_node_coordinates::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree]

    # Stores the quadrature nodes.
    nodes::SVector{NNODES, RealT}

    boundary_names   :: Array{Symbol, 2}      # [face direction, tree]
    current_filename :: String
    unsaved_changes  :: Bool

    max_number_faces :: Int
    ninterfaces      :: Int
    nmortars         :: Int
    nboundaries      :: Int

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
        mesh.unsaved_changes = false # TODO: if set to `true`, the mesh will be saved to a mesh file.
        mesh.tree_node_coordinates = tree_node_coordinates
        mesh.max_number_faces = 4 # TODO: How to automatically adapt for other element types, e.g. triangles?

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
                t8_forest_unref(Ref(mesh.forest))
            end
        end

        # This finalizer call is only recommended during development and not for
        # production runs, especially long-running sessions since a reference to
        # the `mesh` object will be kept throughout the lifetime of the session.
        # See comments in `init_t8code()` in file `src/auxiliary/t8code.jl` for
        # more information.
        if haskey(ENV, "TRIXI_T8CODE_SC_FINALIZE")
            MPI.add_finalize_hook!() do
                t8_forest_unref(Ref(mesh.forest))
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
function T8codeMesh(trees_per_dimension; polydeg = 1,
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
    @assert (NDIMS == 2||NDIMS == 3) "NDIMS should be 2 or 3."

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

    do_partition = 0
    if NDIMS == 2
        conn = T8code.Libt8.p4est_connectivity_new_brick(trees_per_dimension...,
                                                         periodicity...)
        cmesh = t8_cmesh_new_from_p4est(conn, mpi_comm(), do_partition)
        T8code.Libt8.p4est_connectivity_destroy(conn)
    elseif NDIMS == 3
        conn = T8code.Libt8.p8est_connectivity_new_brick(trees_per_dimension...,
                                                         periodicity...)
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

    # Get cell length in reference mesh: Omega_ref = [-1,1]^NDIMS.
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

        # Calculate node coordinates of reference mesh.
        if NDIMS == 2
            cell_x_offset = (verts[1, 1] - 0.5 * (trees_per_dimension[1] - 1)) * dx[1]
            cell_y_offset = (verts[2, 1] - 0.5 * (trees_per_dimension[2] - 1)) * dx[2]

            for j in eachindex(nodes), i in eachindex(nodes)
                tree_node_coordinates[:, i, j, itree] .= mapping_(cell_x_offset +
                                                                  dx[1] * nodes[i] / 2,
                                                                  cell_y_offset +
                                                                  dx[2] * nodes[j] / 2)
            end
        elseif NDIMS == 3
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
    T8codeMesh(cmesh::Ptr{t8_cmesh},
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
function T8codeMesh(cmesh::Ptr{t8_cmesh};
                    mapping = nothing, polydeg = 1, RealT = Float64,
                    initial_refinement_level = 0)
    @assert (t8_cmesh_get_num_trees(cmesh)>0) "Given `cmesh` does not contain any trees."

    # Infer NDIMS from the geometry of the first tree.
    NDIMS = Int(t8_geom_get_dimension(t8_cmesh_get_tree_geometry(cmesh, 0)))

    @assert (NDIMS == 2||NDIMS == 3) "NDIMS should be 2 or 3."

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

    num_local_trees = t8_cmesh_get_num_local_trees(cmesh)

    if NDIMS == 2
        data_in = Array{RealT, 3}(undef, 2, 2, 2)
        tmp1 = zeros(RealT, 2, length(nodes), length(nodes_in))
        verts = zeros(3, 4)

        for itree in 0:(num_local_trees - 1)
            veptr = t8_cmesh_get_tree_vertices(cmesh, itree)

            # Note, `verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))`
            # sometimes does not work since `veptr` is not necessarily properly
            # aligned to 8 bytes.
            for icorner in 1:4
                verts[1, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 1)
                verts[2, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 2)
            end

            # Check if tree's node ordering is right-handed or print a warning.
            let z = zero(eltype(verts)), o = one(eltype(verts))
                u = verts[:, 2] - verts[:, 1]
                v = verts[:, 3] - verts[:, 1]
                w = [z, z, o]

                # Triple product gives signed volume of spanned parallelepiped.
                vol = dot(cross(u, v), w)

                if vol < z
                    @warn "Discovered negative volumes in `cmesh`: vol = $vol"
                end
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
        verts = zeros(3, 8)

        for itree in 0:(num_trees - 1)
            veptr = t8_cmesh_get_tree_vertices(cmesh, itree)

            # Note, `verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))`
            # sometimes does not work since `veptr` is not necessarily properly
            # aligned to 8 bytes.
            for icorner in 1:8
                verts[1, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 1)
                verts[2, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 2)
                verts[3, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 3)
            end

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
    end

    map_node_coordinates!(tree_node_coordinates, mapping)

    # There's no simple and generic way to distinguish boundaries. Name all of them :all.
    boundary_names = fill(:all, 2 * NDIMS, num_trees)

    return T8codeMesh{NDIMS}(forest, tree_node_coordinates, nodes,
                             boundary_names, "")
end

"""
    T8codeMesh(conn::Ptr{p4est_connectivity}; kwargs...)

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

    return T8codeMesh(cmesh; kwargs...)
end

"""
    T8codeMesh(conn::Ptr{p8est_connectivity}; kwargs...)

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

    return T8codeMesh(cmesh; kwargs...)
end

"""
    T8codeMesh(meshfile::String, ndims; kwargs...)

Main mesh constructor for the `T8codeMesh` that imports an unstructured, conforming
mesh from a Gmsh mesh file (`.msh`).

# Arguments
- `meshfile::String`: path to a Gmsh mesh file.
- `ndims`: Mesh file dimension: `2` or `3`.
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
function T8codeMesh(meshfile::String, ndims; kwargs...)
    # Prevent `t8code` from crashing Julia if the file doesn't exist.
    @assert isfile(meshfile)

    meshfile_prefix, meshfile_suffix = splitext(meshfile)

    cmesh = t8_cmesh_from_msh_file(meshfile_prefix, 0, mpi_comm(), ndims, 0, 0)

    return T8codeMesh(cmesh; kwargs...)
end

struct adapt_callback_passthrough
    adapt_callback::Function
    user_data::Any
end

# Callback function prototype to decide for refining and coarsening.
# If `is_family` equals 1, the first `num_elements` in elements
# form a family and we decide whether this family should be coarsened
# or only the first element should be refined.
# Otherwise `is_family` must equal zero and we consider the first entry
# of the element array for refinement.
# Entries of the element array beyond the first `num_elements` are undefined.
# \param [in] forest       the forest to which the new elements belong
# \param [in] forest_from  the forest that is adapted.
# \param [in] which_tree   the local tree containing `elements`
# \param [in] lelement_id  the local element id in `forest_old` in the tree of the current element
# \param [in] ts           the eclass scheme of the tree
# \param [in] is_family    if 1, the first `num_elements` entries in `elements` form a family. If 0, they do not.
# \param [in] num_elements the number of entries in `elements` that are defined
# \param [in] elements     Pointers to a family or, if `is_family` is zero,
#                          pointer to one element.
# \return greater zero if the first entry in `elements` should be refined,
#         smaller zero if the family `elements` shall be coarsened,
#         zero else.
function adapt_callback_wrapper(forest,
                                forest_from,
                                which_tree,
                                lelement_id,
                                ts,
                                is_family,
                                num_elements,
                                elements_ptr)::Cint
    passthrough = unsafe_pointer_to_objref(t8_forest_get_user_data(forest))[]

    elements = unsafe_wrap(Array, elements_ptr, num_elements)

    return passthrough.adapt_callback(forest_from, which_tree, ts, lelement_id, elements,
                                      Bool(is_family), passthrough.user_data)
end

"""
    Trixi.adapt!(mesh::T8codeMesh, adapt_callback; kwargs...)

Adapt a `T8codeMesh` according to a user-defined `adapt_callback`.

# Arguments
- `mesh::T8codeMesh`: Initialized mesh object.
- `adapt_callback`: A user-defined callback which tells the adaption routines
                    if an element should be refined, coarsened or stay unchanged.

    The expected callback signature is as follows:

      `adapt_callback(forest, ltreeid, eclass_scheme, lelemntid, elements, is_family, user_data)`
        # Arguments
        - `forest`: Pointer to the analyzed forest.
        - `ltreeid`: Local index of the current tree where the analyzed elements are part of.
        - `eclass_scheme`: Element class of `elements`.
        - `lelemntid`: Local index of the first element in `elements`.
        - `elements`: Array of elements. If consecutive elements form a family
                      they are passed together, otherwise `elements` consists of just one element.
        - `is_family`: Boolean signifying if `elements` represents a family or not.
        - `user_data`: Void pointer to some arbitrary user data. Default value is `C_NULL`.
        # Returns
          -1 : Coarsen family of elements.
           0 : Stay unchanged.
           1 : Refine element.

- `kwargs`:
    - `recursive = true`: Adapt the forest recursively. If true the caller must ensure that the callback
                          returns 0 for every analyzed element at some point to stop the recursion.
    - `balance = true`: Make sure the adapted forest is 2^(NDIMS-1):1 balanced.
    - `partition = true`: Partition the forest to redistribute elements evenly among MPI ranks.
    - `ghost = true`: Create a ghost layer for MPI data exchange.
    - `user_data = C_NULL`: Pointer to some arbitrary user-defined data.
"""
function adapt!(mesh::T8codeMesh, adapt_callback; recursive = true, balance = true,
                partition = true, ghost = true, user_data = C_NULL)
    # Check that forest is a committed, that is valid and usable, forest.
    @assert t8_forest_is_committed(mesh.forest) != 0

    # Init new forest.
    new_forest_ref = Ref{t8_forest_t}()
    t8_forest_init(new_forest_ref)
    new_forest = new_forest_ref[]

    # Check out `examples/t8_step4_partition_balance_ghost.jl` in
    # https://github.com/DLR-AMR/T8code.jl for detailed explanations.
    let set_from = C_NULL, set_for_coarsening = 0, no_repartition = !partition
        t8_forest_set_user_data(new_forest,
                                pointer_from_objref(Ref(adapt_callback_passthrough(adapt_callback,
                                                                                   user_data))))
        t8_forest_set_adapt(new_forest, mesh.forest,
                            @t8_adapt_callback(adapt_callback_wrapper),
                            recursive)
        if balance
            t8_forest_set_balance(new_forest, set_from, no_repartition)
        end

        if partition
            t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
        end

        t8_forest_set_ghost(new_forest, ghost, T8_GHOST_FACES) # Note: MPI support not available yet so it is a dummy call.

        # The old forest is destroyed here.
        # Call `t8_forest_ref(Ref(mesh.forest))` to keep it.
        t8_forest_commit(new_forest)
    end

    mesh.forest = new_forest

    return nothing
end

"""
    Trixi.balance!(mesh::T8codeMesh)

Balance a `T8codeMesh` to ensure 2^(NDIMS-1):1 face neighbors.
"""
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

"""
    Trixi.partition!(mesh::T8codeMesh)

Partition a `T8codeMesh` in order to redistribute elements evenly among MPI ranks.

# Arguments
- `mesh::T8codeMesh`: Initialized mesh object.
"""
function partition!(mesh::T8codeMesh)
    new_forest_ref = Ref{t8_forest_t}()
    t8_forest_init(new_forest_ref)
    new_forest = new_forest_ref[]

    let set_from = mesh.forest, do_ghost = 1, allow_for_coarsening = 1
        t8_forest_set_partition(new_forest, set_from, allow_for_coarsening)
        t8_forest_set_ghost(new_forest, do_ghost, T8_GHOST_FACES)
        t8_forest_commit(new_forest)
    end

    mesh.forest = new_forest

    return nothing
end

# Compute the global ids (zero-indexed) of first element in each MPI rank.
function get_global_first_element_ids(mesh::T8codeMesh)
    n_elements_local = Int(t8_forest_get_local_num_elements(mesh.forest))
    n_elements_by_rank = Vector{Int}(undef, mpi_nranks())
    n_elements_by_rank[mpi_rank() + 1] = n_elements_local
    MPI.Allgather!(MPI.UBuffer(n_elements_by_rank, 1), mpi_comm())
    return [sum(n_elements_by_rank[1:(rank - 1)]) for rank in 1:(mpi_nranks() + 1)]
end

function count_interfaces(mesh::T8codeMesh)
    @assert t8_forest_is_committed(mesh.forest) != 0

    num_local_elements = t8_forest_get_local_num_elements(mesh.forest)
    num_local_trees = t8_forest_get_num_local_trees(mesh.forest)

    current_index = t8_locidx_t(0)

    local_num_conform = 0
    local_num_mortars = 0
    local_num_boundary = 0

    local_num_mpi_conform = 0
    local_num_mpi_mortars = 0

    visited_global_mortar_ids = Set{UInt64}([])

    max_level = t8_forest_get_maxlevel(mesh.forest) #UInt64
    max_tree_num_elements = UInt64(2^ndims(mesh))^max_level

    if mpi_isparallel()
        ghost_num_trees = t8_forest_ghost_num_trees(mesh.forest)

        ghost_tree_element_offsets = [num_local_elements +
                                      t8_forest_ghost_get_tree_element_offset(mesh.forest,
                                                                              itree)
                                      for itree in 0:(ghost_num_trees - 1)]
        ghost_global_treeids = [t8_forest_ghost_get_global_treeid(mesh.forest, itree)
                                for itree in 0:(ghost_num_trees - 1)]
    end

    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(mesh.forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(mesh.forest, tree_class)

        num_elements_in_tree = t8_forest_get_tree_num_elements(mesh.forest, itree)

        global_itree = t8_forest_global_tree_id(mesh.forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(mesh.forest, itree, ielement)

            level = t8_element_level(eclass_scheme, element)

            num_faces = t8_element_num_faces(eclass_scheme, element)

            # Note: This works only for forests of one element class.
            current_linear_id = global_itree * max_tree_num_elements +
                                t8_element_get_linear_id(eclass_scheme, element, max_level)

            for iface in 0:(num_faces - 1)
                pelement_indices_ref = Ref{Ptr{t8_locidx_t}}()
                pneighbor_leaves_ref = Ref{Ptr{Ptr{t8_element}}}()
                pneigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

                dual_faces_ref = Ref{Ptr{Cint}}()
                num_neighbors_ref = Ref{Cint}()

                forest_is_balanced = Cint(1)

                t8_forest_leaf_face_neighbors(mesh.forest, itree, element,
                                              pneighbor_leaves_ref, iface, dual_faces_ref,
                                              num_neighbors_ref,
                                              pelement_indices_ref, pneigh_scheme_ref,
                                              forest_is_balanced)

                num_neighbors = num_neighbors_ref[]
                dual_faces = unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
                neighbor_ielements = unsafe_wrap(Array, pelement_indices_ref[],
                                                 num_neighbors)
                neighbor_leaves = unsafe_wrap(Array, pneighbor_leaves_ref[], num_neighbors)
                neighbor_scheme = pneigh_scheme_ref[]

                if num_neighbors == 0
                    local_num_boundary += 1
                else
                    neighbor_level = t8_element_level(neighbor_scheme, neighbor_leaves[1])

                    if all(neighbor_ielements .< num_local_elements)
                        # Conforming interface: The second condition ensures we
                        # only visit the interface once.
                        if level == neighbor_level && current_index <= neighbor_ielements[1]
                            local_num_conform += 1
                        elseif level < neighbor_level
                            local_num_mortars += 1
                            # `else level > neighbor_level` is ignored since we
                            # only want to count the mortar interface once.
                        end
                    else
                        if level == neighbor_level
                            local_num_mpi_conform += 1
                        elseif level < neighbor_level
                            local_num_mpi_mortars += 1

                            global_mortar_id = 2 * ndims(mesh) * current_linear_id + iface

                        else # level > neighbor_level
                            neighbor_global_ghost_itree = ghost_global_treeids[findlast(ghost_tree_element_offsets .<=
                                                                                        neighbor_ielements[1])]
                            neighbor_linear_id = neighbor_global_ghost_itree *
                                                 max_tree_num_elements +
                                                 t8_element_get_linear_id(neighbor_scheme,
                                                                          neighbor_leaves[1],
                                                                          max_level)
                            global_mortar_id = 2 * ndims(mesh) * neighbor_linear_id +
                                               dual_faces[1]

                            if !(global_mortar_id in visited_global_mortar_ids)
                                push!(visited_global_mortar_ids, global_mortar_id)
                                local_num_mpi_mortars += 1
                            end
                        end
                    end
                end

                t8_free(dual_faces_ref[])
                t8_free(pneighbor_leaves_ref[])
                t8_free(pelement_indices_ref[])
            end # for

            current_index += 1
        end # for
    end # for

    return (interfaces = local_num_conform,
            mortars = local_num_mortars,
            boundaries = local_num_boundary,
            mpi_interfaces = local_num_mpi_conform,
            mpi_mortars = local_num_mpi_mortars)
end

# I know this routine is an unmaintainable behemoth. However, I see no real
# and elegant way to refactor this into, for example, smaller parts. The
# `t8_forest_leaf_face_neighbors` routine is as of now rather costly and it
# makes sense to query it only once per face per element and extract all the
# information needed at once in order to fill the connectivity information.
# Instead, I opted for good documentation.
function fill_mesh_info!(mesh::T8codeMesh, interfaces, mortars, boundaries,
                         boundary_names; mpi_mesh_info = nothing)
    @assert t8_forest_is_committed(mesh.forest) != 0

    num_local_elements = t8_forest_get_local_num_elements(mesh.forest)
    num_local_trees = t8_forest_get_num_local_trees(mesh.forest)

    if !isnothing(mpi_mesh_info)
        #! format: off
        remotes = t8_forest_ghost_get_remotes(mesh.forest)
        ghost_num_trees = t8_forest_ghost_num_trees(mesh.forest)

        ghost_remote_first_elem = [num_local_elements +
                                   t8_forest_ghost_remote_first_elem(mesh.forest, remote)
                                   for remote in remotes]

        ghost_tree_element_offsets = [num_local_elements +
                                      t8_forest_ghost_get_tree_element_offset(mesh.forest, itree)
                                      for itree in 0:(ghost_num_trees - 1)]

        ghost_global_treeids = [t8_forest_ghost_get_global_treeid(mesh.forest, itree)
                                for itree in 0:(ghost_num_trees - 1)]
        #! format: on
    end

    # Process-local index of the current element in the space-filling curve.
    current_index = t8_locidx_t(0)

    # Increment counters for the different interface/mortar/boundary types.
    local_num_conform = 0
    local_num_mortars = 0
    local_num_boundary = 0

    local_num_mpi_conform = 0
    local_num_mpi_mortars = 0

    # Works for quads and hexs only. This mapping is needed in the MPI mortar
    # sections below.
    map_iface_to_ichild_to_position = [
        # 0  1  2  3  4  5  6  7 ichild/iface
        [1, 0, 2, 0, 3, 0, 4, 0], # 0
        [0, 1, 0, 2, 0, 3, 0, 4], # 1
        [1, 2, 0, 0, 3, 4, 0, 0], # 2
        [0, 0, 1, 2, 0, 0, 3, 4], # 3
        [1, 2, 3, 4, 0, 0, 0, 0], # 4
        [0, 0, 0, 0, 1, 2, 3, 4], # 5
    ]

    # Helper variables to compute unique global MPI interface/mortar ids.
    max_level = t8_forest_get_maxlevel(mesh.forest) #UInt64
    max_tree_num_elements = UInt64(2^ndims(mesh))^max_level

    # These two variables help to ensure that we count MPI mortars from smaller
    # elements point of view only once.
    visited_global_mortar_ids = Set{UInt64}([])
    global_mortar_id_to_local = Dict{UInt64, Int}([])

    # Loop over all local trees.
    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(mesh.forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(mesh.forest, tree_class)

        num_elements_in_tree = t8_forest_get_tree_num_elements(mesh.forest, itree)

        global_itree = t8_forest_global_tree_id(mesh.forest, itree)

        # Loop over all local elements of the current local tree.
        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(mesh.forest, itree, ielement)

            level = t8_element_level(eclass_scheme, element)

            num_faces = t8_element_num_faces(eclass_scheme, element)

            # Note: This works only for forests of one element class.
            current_linear_id = global_itree * max_tree_num_elements +
                                t8_element_get_linear_id(eclass_scheme, element, max_level)

            # Loop over all faces of the current local element.
            for iface in 0:(num_faces - 1)
                # Compute the `orientation` of the touching faces.
                if t8_element_is_root_boundary(eclass_scheme, element, iface) == 1
                    cmesh = t8_forest_get_cmesh(mesh.forest)
                    itree_in_cmesh = t8_forest_ltreeid_to_cmesh_ltreeid(mesh.forest, itree)
                    iface_in_tree = t8_element_tree_face(eclass_scheme, element, iface)
                    orientation_ref = Ref{Cint}()

                    t8_cmesh_get_face_neighbor(cmesh, itree_in_cmesh, iface_in_tree, C_NULL,
                                               orientation_ref)
                    orientation = orientation_ref[]
                else
                    orientation = zero(Cint)
                end

                pelement_indices_ref = Ref{Ptr{t8_locidx_t}}()
                pneighbor_leaves_ref = Ref{Ptr{Ptr{t8_element}}}()
                pneigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

                dual_faces_ref = Ref{Ptr{Cint}}()
                num_neighbors_ref = Ref{Cint}()

                forest_is_balanced = Cint(1)

                # Query neighbor information from t8code.
                t8_forest_leaf_face_neighbors(mesh.forest, itree, element,
                                              pneighbor_leaves_ref, iface, dual_faces_ref,
                                              num_neighbors_ref,
                                              pelement_indices_ref, pneigh_scheme_ref,
                                              forest_is_balanced)

                num_neighbors = num_neighbors_ref[]
                dual_faces = unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
                neighbor_ielements = unsafe_wrap(Array, pelement_indices_ref[],
                                                 num_neighbors)
                neighbor_leaves = unsafe_wrap(Array, pneighbor_leaves_ref[], num_neighbors)
                neighbor_scheme = pneigh_scheme_ref[]

                # Now we check for the different cases. The nested if-structure is as follows:
                #
                #   if `boundary`:
                #     <fill boundary info>
                #
                #   else: // It must be an interface or mortar.
                #
                #     if `all neighbors are local elements`:
                #
                #       if `local interface`:
                #         <fill interface info>
                #       elseif `local mortar from larger element point of view`:
                #         <fill mortar info>
                #       else: // `local mortar from smaller elements point of view`
                #         <skip> // We only count local mortars once.
                #
                #     else: // It must be either a MPI interface or a MPI mortar.
                #
                #       if `MPI interface`:
                #         <fill MPI interface info>
                #       elseif `MPI mortar from larger element point of view`:
                #         <fill MPI mortar info>
                #       else: // `MPI mortar from smaller elements point of view`
                #         <fill MPI mortar info>
                #
                #   // end

                # Domain boundary.
                if num_neighbors == 0
                    local_num_boundary += 1
                    boundary_id = local_num_boundary

                    boundaries.neighbor_ids[boundary_id] = current_index + 1

                    init_boundary_node_indices!(boundaries, iface, boundary_id)

                    # One-based indexing.
                    boundaries.name[boundary_id] = boundary_names[iface + 1, itree + 1]

                    # Interface or mortar.
                else
                    neighbor_level = t8_element_level(neighbor_scheme, neighbor_leaves[1])

                    # Local interface or mortar.
                    if all(neighbor_ielements .< num_local_elements)

                        # Local interface: The second condition ensures we only visit the interface once.
                        if level == neighbor_level && current_index <= neighbor_ielements[1]
                            local_num_conform += 1

                            interfaces.neighbor_ids[1, local_num_conform] = current_index +
                                                                            1
                            interfaces.neighbor_ids[2, local_num_conform] = neighbor_ielements[1] +
                                                                            1

                            init_interface_node_indices!(interfaces, (iface, dual_faces[1]),
                                                         orientation,
                                                         local_num_conform)
                            # Local mortar.
                        elseif level < neighbor_level
                            local_num_mortars += 1

                            # Last entry is the large element.
                            mortars.neighbor_ids[end, local_num_mortars] = current_index + 1

                            init_mortar_neighbor_ids!(mortars, iface, dual_faces[1],
                                                      orientation, neighbor_ielements,
                                                      local_num_mortars)

                            init_mortar_node_indices!(mortars, (dual_faces[1], iface),
                                                      orientation, local_num_mortars)

                            # else: `level > neighbor_level` is skipped since we visit the mortar interface only once.
                        end

                        # MPI interface or MPI mortar.
                    else

                        # MPI interface.
                        if level == neighbor_level
                            local_num_mpi_conform += 1

                            neighbor_global_ghost_itree = ghost_global_treeids[findlast(ghost_tree_element_offsets .<=
                                                                                        neighbor_ielements[1])]

                            neighbor_linear_id = neighbor_global_ghost_itree *
                                                 max_tree_num_elements +
                                                 t8_element_get_linear_id(neighbor_scheme,
                                                                          neighbor_leaves[1],
                                                                          max_level)

                            if current_linear_id < neighbor_linear_id
                                local_side = 1
                                smaller_iface = iface
                                smaller_linear_id = current_linear_id
                                faces = (iface, dual_faces[1])
                            else
                                local_side = 2
                                smaller_iface = dual_faces[1]
                                smaller_linear_id = neighbor_linear_id
                                faces = (dual_faces[1], iface)
                            end

                            global_interface_id = 2 * ndims(mesh) * smaller_linear_id +
                                                  smaller_iface

                            mpi_mesh_info.mpi_interfaces.local_neighbor_ids[local_num_mpi_conform] = current_index +
                                                                                                     1
                            mpi_mesh_info.mpi_interfaces.local_sides[local_num_mpi_conform] = local_side

                            init_mpi_interface_node_indices!(mpi_mesh_info.mpi_interfaces,
                                                             faces, local_side, orientation,
                                                             local_num_mpi_conform)

                            neighbor_rank = remotes[findlast(ghost_remote_first_elem .<=
                                                             neighbor_ielements[1])]
                            mpi_mesh_info.neighbor_ranks_interface[local_num_mpi_conform] = neighbor_rank

                            mpi_mesh_info.global_interface_ids[local_num_mpi_conform] = global_interface_id

                            # MPI Mortar: from larger element point of view
                        elseif level < neighbor_level
                            local_num_mpi_mortars += 1

                            global_mortar_id = 2 * ndims(mesh) * current_linear_id + iface

                            neighbor_ids = neighbor_ielements .+ 1

                            local_neighbor_positions = findall(neighbor_ids .<=
                                                               num_local_elements)
                            local_neighbor_ids = [neighbor_ids[i]
                                                  for i in local_neighbor_positions]
                            local_neighbor_positions = [map_iface_to_ichild_to_position[dual_faces[1] + 1][t8_element_child_id(neighbor_scheme, neighbor_leaves[i]) + 1]
                                                        for i in local_neighbor_positions]

                            # Last entry is the large element.
                            push!(local_neighbor_ids, current_index + 1)
                            push!(local_neighbor_positions, 2^(ndims(mesh) - 1) + 1)

                            mpi_mesh_info.mpi_mortars.local_neighbor_ids[local_num_mpi_mortars] = local_neighbor_ids
                            mpi_mesh_info.mpi_mortars.local_neighbor_positions[local_num_mpi_mortars] = local_neighbor_positions

                            init_mortar_node_indices!(mpi_mesh_info.mpi_mortars,
                                                      (dual_faces[1], iface), orientation,
                                                      local_num_mpi_mortars)

                            neighbor_ranks = [remotes[findlast(ghost_remote_first_elem .<=
                                                               ineighbor_ghost)]
                                              for ineighbor_ghost in filter(x -> x >=
                                                                                 num_local_elements,
                                                                            neighbor_ielements)]
                            mpi_mesh_info.neighbor_ranks_mortar[local_num_mpi_mortars] = neighbor_ranks

                            mpi_mesh_info.global_mortar_ids[local_num_mpi_mortars] = global_mortar_id

                            # MPI Mortar: from smaller elements point of view
                        else
                            neighbor_global_ghost_itree = ghost_global_treeids[findlast(ghost_tree_element_offsets .<=
                                                                                        neighbor_ielements[1])]
                            neighbor_linear_id = neighbor_global_ghost_itree *
                                                 max_tree_num_elements +
                                                 t8_element_get_linear_id(neighbor_scheme,
                                                                          neighbor_leaves[1],
                                                                          max_level)
                            global_mortar_id = 2 * ndims(mesh) * neighbor_linear_id +
                                               dual_faces[1]

                            if global_mortar_id in visited_global_mortar_ids
                                local_mpi_mortar_id = global_mortar_id_to_local[global_mortar_id]

                                push!(mpi_mesh_info.mpi_mortars.local_neighbor_ids[local_mpi_mortar_id],
                                      current_index + 1)
                                push!(mpi_mesh_info.mpi_mortars.local_neighbor_positions[local_mpi_mortar_id],
                                      map_iface_to_ichild_to_position[iface + 1][t8_element_child_id(eclass_scheme, element) + 1])
                            else
                                local_num_mpi_mortars += 1
                                local_mpi_mortar_id = local_num_mpi_mortars
                                push!(visited_global_mortar_ids, global_mortar_id)
                                global_mortar_id_to_local[global_mortar_id] = local_mpi_mortar_id

                                mpi_mesh_info.mpi_mortars.local_neighbor_ids[local_mpi_mortar_id] = [
                                    current_index + 1,
                                ]
                                mpi_mesh_info.mpi_mortars.local_neighbor_positions[local_mpi_mortar_id] = [
                                    map_iface_to_ichild_to_position[iface + 1][t8_element_child_id(eclass_scheme, element) + 1],
                                ]
                                init_mortar_node_indices!(mpi_mesh_info.mpi_mortars,
                                                          (iface, dual_faces[1]),
                                                          orientation, local_mpi_mortar_id)

                                neighbor_ranks = [
                                    remotes[findlast(ghost_remote_first_elem .<=
                                                     neighbor_ielements[1])],
                                ]
                                mpi_mesh_info.neighbor_ranks_mortar[local_mpi_mortar_id] = neighbor_ranks

                                mpi_mesh_info.global_mortar_ids[local_mpi_mortar_id] = global_mortar_id
                            end
                        end
                    end
                end

                t8_free(dual_faces_ref[])
                t8_free(pneighbor_leaves_ref[])
                t8_free(pelement_indices_ref[])
            end # for iface

            current_index += 1
        end # for ielement
    end # for itree

    return nothing
end

#! format: off
@deprecate T8codeMesh{2}(conn::Ptr{p4est_connectivity}; kwargs...) T8codeMesh(conn::Ptr{p4est_connectivity}; kwargs...)
@deprecate T8codeMesh{3}(conn::Ptr{p8est_connectivity}; kwargs...) T8codeMesh(conn::Ptr{p8est_connectivity}; kwargs...)
@deprecate T8codeMesh{2}(meshfile::String; kwargs...) T8codeMesh(meshfile::String, 2; kwargs...)
@deprecate T8codeMesh{3}(meshfile::String; kwargs...) T8codeMesh(meshfile::String, 3; kwargs...)
#! format: on

# Write the forest as vtu and also write the element's volumes in the file.
#
# t8code supports writing element based data to vtu as long as its stored
# as doubles. Each of the data fields to write has to be provided in its own
# array of length num_local_elements.
# t8code supports two types: T8_VTK_SCALAR - One double per element.
#                       and  T8_VTK_VECTOR - Three doubles per element.
function output_data_to_vtu(mesh::T8codeMesh, equations, solver,
                            u_tmp, out)
    vars = varnames(cons2cons, equations)

    vtk_data = Vector{t8_vtk_data_field_t}(undef, nvariables(equations))

    data = Array{Float64}(undef, ncells(mesh), nvariables(equations))
    for v in eachvariable(equations)
        for element in 1:ncells(mesh)
            data[element, v] = u_tmp[element].u[v]
        end
    end

    GC.@preserve data begin
        for v in eachvariable(equations)
            data_ptr = pointer(@views(data[:, v]))
            vtk_data[v] = t8_vtk_data_field_t(T8_VTK_SCALAR,
                                              NTuple{8192, Cchar}(rpad("$(vars[v])\0",
                                                                       8192, ' ')),
                                              data_ptr)
        end

        # The number of user defined data fields to write.
        num_data = length(vtk_data)

        # Write user defined data to vtu file.
        write_treeid = 1
        write_mpirank = 1
        write_level = 1
        write_element_id = 1
        write_ghosts = 0
        t8_forest_write_vtk_ext(mesh.forest, out, write_treeid, write_mpirank,
                                write_level, write_element_id, write_ghosts,
                                0, 0, num_data, pointer(vtk_data))
    end
end

# Simple meshes
# Temporary routines to create simple `cmesh`s by hand

# Directly ported from: `src/t8_cmesh/t8_cmesh_examples.c: t8_cmesh_new_periodic_hybrid`.
function cmesh_new_periodic_hybrid(comm)::t8_cmesh_t
    n_dims = 2
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0, # tree 0, triangle
        0, -1.0, 0,
        0, 0, 0,
        -1.0, -1.0, 0, # tree 1, triangle
        0, 0, 0,
        -1.0, 0, 0,
        0, -1.0, 0,    # tree 2, quad
        1.0, -1.0, 0,
        0, 0, 0,
        1.0, 0, 0,
        -1.0, 0, 0,    # tree 3, quad
        0, 0, 0,
        -1.0, 1.0, 0,
        0, 1.0, 0,
        0, 0, 0,       # tree 4, triangle
        1.0, 0, 0,
        1.0, 1.0, 0,
        0, 0, 0,       # tree 5, triangle
        1.0, 1.0, 0,
        0, 1.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    # This is how the cmesh looks like. The numbers are the tree numbers:
    # Domain size [-1,1]^2
    #
    #   +---+---+
    #   |   |5 /|
    #   | 3 | / |
    #   |   |/ 4|
    #   +---+---+
    #   |1 /|   |
    #   | / | 2 |
    #   |/0 |   |
    #   +---+---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_QUAD)
    t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_QUAD)
    t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 5, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[(1 + 9):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[(1 + 18):end]), 4)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[(1 + 30):end]), 4)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[(1 + 42):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 5, @views(vertices[(1 + 51):end]), 3)

    t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 0, 2, 0, 0, 0)
    t8_cmesh_set_join(cmesh, 0, 3, 2, 3, 0)

    t8_cmesh_set_join(cmesh, 1, 3, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 1, 2, 1, 1, 0)

    t8_cmesh_set_join(cmesh, 2, 4, 3, 2, 0)
    t8_cmesh_set_join(cmesh, 2, 5, 2, 0, 1)

    t8_cmesh_set_join(cmesh, 3, 5, 1, 1, 0)
    t8_cmesh_set_join(cmesh, 3, 4, 0, 0, 0)

    t8_cmesh_set_join(cmesh, 4, 5, 1, 2, 0)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_quad_nonperiodic(comm)::t8_cmesh_t
    n_dims = 2
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0, # tree 0, quad
        1.0, -1.0, 0,
        -1.0, 1.0, 0,
        1.0, 1.0, 0,

        # rotated:
        # -1.0, 0.0, 0,  # tree 0, quad
        # 0.0, -1.0, 0,
        # 0.0, 1.0, 0,
        # 1.0, 0.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    # This is how the cmesh looks like. The numbers are the tree numbers:
    #
    #   +---+
    #   |   |
    #   | 0 |
    #   |   |
    #   +---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_QUAD)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 4)

    # t8_cmesh_set_join(cmesh, 0, 0, 0, 1, 0)
    # t8_cmesh_set_join(cmesh, 0, 0, 2, 3, 0)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_quad(comm)::t8_cmesh_t
    n_dims = 2
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0, # tree 0, quad
        1.0, -1.0, 0,
        -1.0, 1.0, 0,
        1.0, 1.0, 0,

        # rotated:
        # -1.0, 0.0, 0,  # tree 0, quad
        # 0.0, -1.0, 0,
        # 0.0, 1.0, 0,
        # 1.0, 0.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    # This is how the cmesh looks like. The numbers are the tree numbers:
    #
    #   +---+
    #   |   |
    #   | 0 |
    #   |   |
    #   +---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_QUAD)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 4)

    t8_cmesh_set_join(cmesh, 0, 0, 0, 1, 0)
    t8_cmesh_set_join(cmesh, 0, 0, 2, 3, 0)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_tri(comm)::t8_cmesh_t
    n_dims = 2
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0, # tree 0, triangle
        1.0, -1.0, 0,
        1.0, 1.0, 0,
        -1.0, -1.0, 0, # tree 1, triangle
        1.0, 1.0, 0,
        -1.0, 1.0, 0,

        # rotated:
        # -1.0, 0, 0,  # tree 0, triangle
        # 0, -1.0, 0,
        # 1.0, 0, 0,
        # -1.0, 0, 0,  # tree 1, triangle
        # 1.0, 0, 0,
        # 0, 1.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    # This is how the cmesh looks like. The numbers are the tree numbers:
    #
    #   +---+
    #   |1 /|
    #   | / |
    #   |/0 |
    #   +---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[(1 + 9):end]), 3)

    t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 0, 1, 0, 1, 0)
    t8_cmesh_set_join(cmesh, 0, 1, 2, 0, 1)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_tri2(comm)::t8_cmesh_t
    n_dims = 2
    vertices = [ # Just all vertices of all trees. partly duplicated
        -1.0, -1.0, 0,  # tree 0, triangle
        0, -1.0, 0,
        0, 0, 0,
        -1.0, -1.0, 0,  # tree 1, triangle
        0, 0, 0,
        -1.0, 0, 0,
        0, -1.0, 0,     # tree 2, triangle
        1.0, -1.0, 0,
        1.0, 0, 0,
        0, -1.0, 0,     # tree 3, triangle
        1.0, 0, 0,
        0, 0, 0,
        -1.0, 0, 0,     # tree 4, triangle
        0, 0, 0,
        -1.0, 1.0, 0,
        -1.0, 1.0, 0,   # tree 5, triangle
        0, 0, 0,
        0, 1.0, 0,
        0, 0, 0,        # tree 6, triangle
        1.0, 0, 0,
        0, 1.0, 0,
        0, 1.0, 0,      # tree 7, triangle
        1.0, 0, 0,
        1.0, 1.0, 0,
    ]

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    #
    # This is how the cmesh looks like. The numbers are the tree numbers:
    #
    #   +---+---+
    #   |\ 5|\ 7|
    #   | \ | \ |
    #   |4 \| 6\|
    #   +---+---+
    #   |1 /|3 /|
    #   | / | / |
    #   |/0 |/ 2|
    #   +---+---+
    #

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 5, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 6, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 7, T8_ECLASS_TRIANGLE)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[(1 + 9):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[(1 + 18):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[(1 + 27):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[(1 + 36):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 5, @views(vertices[(1 + 45):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 6, @views(vertices[(1 + 54):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 7, @views(vertices[(1 + 63):end]), 3)

    t8_cmesh_set_join(cmesh, 0, 1, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 0, 3, 0, 1, 0)
    t8_cmesh_set_join(cmesh, 0, 5, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 1, 4, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 1, 2, 1, 0, 0)

    t8_cmesh_set_join(cmesh, 2, 3, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 2, 7, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 3, 6, 0, 2, 1)

    t8_cmesh_set_join(cmesh, 4, 5, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 4, 7, 1, 0, 0)

    t8_cmesh_set_join(cmesh, 5, 6, 0, 1, 0)

    t8_cmesh_set_join(cmesh, 6, 7, 0, 2, 1)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end

function cmesh_new_periodic_hybrid2(comm)::t8_cmesh_t
    n_dims = 2
    vertices = [  # Just all vertices of all trees. partly duplicated
        -2.0, -2.0, 0,  # tree 0, triangle
        0, -2.0, 0,
        -2.0, 0, 0,
        -2.0, 2.0, 0,   # tree 1, triangle
        -2.0, 0, 0,
        0, 2.0, 0,
        2.0, -2.0, 0,   # tree 2, triangle
        2.0, 0, 0,
        0, -2.0, 0,
        2.0, 2.0, 0,    # tree 3, triangle
        0, 2.0, 0,
        2.0, 0, 0,
        0, -2.0, 0,     # tree 4, quad
        2.0, 0, 0,
        -2.0, 0, 0,
        0, 2.0, 0,
    ]

    # This is how the cmesh looks like. The numbers are the tree numbers:
    # Domain size [-2,2]^2
    #
    # +----------+
    # | 1  /\  3 |
    # |   /  \   |
    # |  /    \  |
    # | /      \ |
    # |/   4    \|
    # |\        /|
    # | \      / |
    # |  \    /  |
    # | 0 \  / 2 |
    # |    \/    |
    # +----------+
    #

    # Generally, one can define other geometries. But besides linear the other
    # geometries in t8code do not have C interface yet.
    linear_geom = t8_geometry_linear_new(n_dims)

    cmesh_ref = Ref(t8_cmesh_t())
    t8_cmesh_init(cmesh_ref)
    cmesh = cmesh_ref[]

    # Use linear geometry
    t8_cmesh_register_geometry(cmesh, linear_geom)
    t8_cmesh_set_tree_class(cmesh, 0, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 1, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 2, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 3, T8_ECLASS_TRIANGLE)
    t8_cmesh_set_tree_class(cmesh, 4, T8_ECLASS_QUAD)

    t8_cmesh_set_tree_vertices(cmesh, 0, @views(vertices[(1 + 0):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 1, @views(vertices[(1 + 9):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 2, @views(vertices[(1 + 18):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 3, @views(vertices[(1 + 27):end]), 3)
    t8_cmesh_set_tree_vertices(cmesh, 4, @views(vertices[(1 + 36):end]), 4)

    t8_cmesh_set_join(cmesh, 0, 4, 0, 0, 0)
    t8_cmesh_set_join(cmesh, 0, 2, 1, 2, 0)
    t8_cmesh_set_join(cmesh, 0, 1, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 1, 4, 0, 3, 0)
    t8_cmesh_set_join(cmesh, 1, 3, 2, 1, 0)

    t8_cmesh_set_join(cmesh, 2, 4, 0, 2, 1)
    t8_cmesh_set_join(cmesh, 2, 3, 1, 2, 0)

    t8_cmesh_set_join(cmesh, 3, 4, 0, 1, 1)

    t8_cmesh_commit(cmesh, comm)

    return cmesh
end
