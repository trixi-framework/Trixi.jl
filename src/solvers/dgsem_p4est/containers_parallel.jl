# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct P4estMPIInterfaceContainer{NDIMS, uEltype <: Real, NDIMSP2} <:
               AbstractContainer
    u::Array{uEltype, NDIMSP2}                  # [primary/secondary, variable, i, j, interface]
    local_neighbor_ids::Vector{Int}             # [interface]
    node_indices::Vector{NTuple{NDIMS, Symbol}} # [interface]
    local_sides::Vector{Int}                    # [interface]

    # internal `resize!`able storage
    _u::Vector{uEltype}
end

@inline function nmpiinterfaces(interfaces::P4estMPIInterfaceContainer)
    length(interfaces.local_sides)
end
@inline Base.ndims(::P4estMPIInterfaceContainer{NDIMS}) where {NDIMS} = NDIMS

function Base.resize!(mpi_interfaces::P4estMPIInterfaceContainer, capacity)
    @unpack _u, local_neighbor_ids, node_indices, local_sides = mpi_interfaces

    n_dims = ndims(mpi_interfaces)
    n_nodes = size(mpi_interfaces.u, 3)
    n_variables = size(mpi_interfaces.u, 2)

    resize!(_u, 2 * n_variables * n_nodes^(n_dims - 1) * capacity)
    mpi_interfaces.u = unsafe_wrap(Array, pointer(_u),
                                   (2, n_variables, ntuple(_ -> n_nodes, n_dims - 1)...,
                                    capacity))

    resize!(local_neighbor_ids, capacity)

    resize!(node_indices, capacity)

    resize!(local_sides, capacity)

    return nothing
end

# Create MPI interface container and initialize interface data
function init_mpi_interfaces(mesh::Union{ParallelP4estMesh, ParallelT8codeMesh},
                             equations, basis, elements)
    NDIMS = ndims(elements)
    uEltype = eltype(elements)

    # Initialize container
    n_mpi_interfaces = count_required_surfaces(mesh).mpi_interfaces

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * nnodes(basis)^(NDIMS - 1) *
                         n_mpi_interfaces)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                     n_mpi_interfaces))

    local_neighbor_ids = Vector{Int}(undef, n_mpi_interfaces)

    node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, n_mpi_interfaces)

    local_sides = Vector{Int}(undef, n_mpi_interfaces)

    mpi_interfaces = P4estMPIInterfaceContainer{NDIMS, uEltype, NDIMS + 2}(u,
                                                                           local_neighbor_ids,
                                                                           node_indices,
                                                                           local_sides,
                                                                           _u)

    init_mpi_interfaces!(mpi_interfaces, mesh)

    return mpi_interfaces
end

function init_mpi_interfaces!(mpi_interfaces, mesh::ParallelP4estMesh)
    init_surfaces!(nothing, nothing, nothing, mpi_interfaces, nothing, mesh)

    return mpi_interfaces
end

# Container data structure (structure-of-arrays style) for DG L2 mortars
#
# Similar to `P4estMortarContainer`. The field `neighbor_ids` has been split up into
# `local_neighbor_ids` and `local_neighbor_positions` to describe the ids and positions of the locally
# available elements belonging to a particular MPI mortar. Furthermore, `normal_directions` holds
# the normal vectors on the surface of the small elements for each mortar.
mutable struct P4estMPIMortarContainer{NDIMS, uEltype <: Real, RealT <: Real, NDIMSP1,
                                       NDIMSP2, NDIMSP3} <: AbstractContainer
    u::Array{uEltype, NDIMSP3}                    # [small/large side, variable, position, i, j, mortar]
    local_neighbor_ids::Vector{Vector{Int}}       # [mortar][ids]
    local_neighbor_positions::Vector{Vector{Int}} # [mortar][positions]
    node_indices::Matrix{NTuple{NDIMS, Symbol}}   # [small/large, mortar]
    normal_directions::Array{RealT, NDIMSP2}      # [dimension, i, j, position, mortar]
    # internal `resize!`able storage
    _u::Vector{uEltype}
    _node_indices::Vector{NTuple{NDIMS, Symbol}}
    _normal_directions::Vector{RealT}
end

@inline function nmpimortars(mpi_mortars::P4estMPIMortarContainer)
    length(mpi_mortars.local_neighbor_ids)
end
@inline Base.ndims(::P4estMPIMortarContainer{NDIMS}) where {NDIMS} = NDIMS

function Base.resize!(mpi_mortars::P4estMPIMortarContainer, capacity)
    @unpack _u, _node_indices, _normal_directions = mpi_mortars

    n_dims = ndims(mpi_mortars)
    n_nodes = size(mpi_mortars.u, 4)
    n_variables = size(mpi_mortars.u, 2)

    resize!(_u, 2 * n_variables * 2^(n_dims - 1) * n_nodes^(n_dims - 1) * capacity)
    mpi_mortars.u = unsafe_wrap(Array, pointer(_u),
                                (2, n_variables, 2^(n_dims - 1),
                                 ntuple(_ -> n_nodes, n_dims - 1)..., capacity))

    resize!(mpi_mortars.local_neighbor_ids, capacity)
    resize!(mpi_mortars.local_neighbor_positions, capacity)

    resize!(_node_indices, 2 * capacity)
    mpi_mortars.node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, capacity))

    resize!(_normal_directions,
            n_dims * n_nodes^(n_dims - 1) * 2^(n_dims - 1) * capacity)
    mpi_mortars.normal_directions = unsafe_wrap(Array, pointer(_normal_directions),
                                                (n_dims,
                                                 ntuple(_ -> n_nodes, n_dims - 1)...,
                                                 2^(n_dims - 1), capacity))

    return nothing
end

# Create MPI mortar container and initialize MPI mortar data
function init_mpi_mortars(mesh::Union{ParallelP4estMesh, ParallelT8codeMesh}, equations,
                          basis, elements)
    NDIMS = ndims(mesh)
    RealT = real(mesh)
    uEltype = eltype(elements)

    # Initialize container
    n_mpi_mortars = count_required_surfaces(mesh).mpi_mortars

    _u = Vector{uEltype}(undef,
                         2 * nvariables(equations) * 2^(NDIMS - 1) *
                         nnodes(basis)^(NDIMS - 1) * n_mpi_mortars)
    u = unsafe_wrap(Array, pointer(_u),
                    (2, nvariables(equations), 2^(NDIMS - 1),
                     ntuple(_ -> nnodes(basis), NDIMS - 1)..., n_mpi_mortars))

    local_neighbor_ids = fill(Vector{Int}(), n_mpi_mortars)
    local_neighbor_positions = fill(Vector{Int}(), n_mpi_mortars)

    _node_indices = Vector{NTuple{NDIMS, Symbol}}(undef, 2 * n_mpi_mortars)
    node_indices = unsafe_wrap(Array, pointer(_node_indices), (2, n_mpi_mortars))

    _normal_directions = Vector{RealT}(undef,
                                       NDIMS * nnodes(basis)^(NDIMS - 1) *
                                       2^(NDIMS - 1) * n_mpi_mortars)
    normal_directions = unsafe_wrap(Array, pointer(_normal_directions),
                                    (NDIMS, ntuple(_ -> nnodes(basis), NDIMS - 1)...,
                                     2^(NDIMS - 1), n_mpi_mortars))

    mpi_mortars = P4estMPIMortarContainer{NDIMS, uEltype, RealT, NDIMS + 1, NDIMS + 2,
                                          NDIMS + 3}(u, local_neighbor_ids,
                                                     local_neighbor_positions,
                                                     node_indices, normal_directions,
                                                     _u, _node_indices,
                                                     _normal_directions)

    if n_mpi_mortars > 0
        init_mpi_mortars!(mpi_mortars, mesh, basis, elements)
    end

    return mpi_mortars
end

function init_mpi_mortars!(mpi_mortars, mesh::ParallelP4estMesh, basis, elements)
    init_surfaces!(nothing, nothing, nothing, nothing, mpi_mortars, mesh)
    init_normal_directions!(mpi_mortars, basis, elements)

    return mpi_mortars
end

# Overload init! function for regular interfaces, regular mortars and boundaries since they must
# call the appropriate init_surfaces! function for parallel p4est meshes
function init_interfaces!(interfaces, mesh::ParallelP4estMesh)
    init_surfaces!(interfaces, nothing, nothing, nothing, nothing, mesh)

    return interfaces
end

function init_mortars!(mortars, mesh::ParallelP4estMesh)
    init_surfaces!(nothing, mortars, nothing, nothing, nothing, mesh)

    return mortars
end

function init_boundaries!(boundaries, mesh::ParallelP4estMesh)
    init_surfaces!(nothing, nothing, boundaries, nothing, nothing, mesh)

    return boundaries
end

function reinitialize_containers!(mesh::ParallelP4estMesh, equations, dg::DGSEM, cache)
    # Make sure to re-create ghost layer before reinitializing MPI-related containers
    update_ghost_layer!(mesh)

    # Re-initialize elements container
    @unpack elements = cache
    resize!(elements, ncells(mesh))
    init_elements!(elements, mesh, dg.basis)

    required = count_required_surfaces(mesh)

    # resize interfaces container
    @unpack interfaces = cache
    resize!(interfaces, required.interfaces)

    # resize boundaries container
    @unpack boundaries = cache
    resize!(boundaries, required.boundaries)

    # resize mortars container
    @unpack mortars = cache
    resize!(mortars, required.mortars)

    # resize mpi_interfaces container
    @unpack mpi_interfaces = cache
    resize!(mpi_interfaces, required.mpi_interfaces)

    # resize mpi_mortars container
    @unpack mpi_mortars = cache
    resize!(mpi_mortars, required.mpi_mortars)

    # re-initialize containers together to reduce
    # the number of iterations over the mesh in p4est
    init_surfaces!(interfaces, mortars, boundaries, mpi_interfaces, mpi_mortars, mesh)

    # re-initialize MPI cache
    @unpack mpi_cache = cache
    init_mpi_cache!(mpi_cache, mesh, mpi_interfaces, mpi_mortars,
                    nvariables(equations), nnodes(dg), eltype(elements))

    # re-initialize and distribute normal directions of MPI mortars; requires MPI communication, so
    # the MPI cache must be re-initialized before
    init_normal_directions!(mpi_mortars, dg.basis, elements)
    exchange_normal_directions!(mpi_mortars, mpi_cache, mesh, nnodes(dg))
end

# A helper struct used in initialization methods below
mutable struct ParallelInitSurfacesIterFaceUserData{Interfaces, Mortars, Boundaries,
                                                    MPIInterfaces, MPIMortars, Mesh}
    interfaces::Interfaces
    interface_id::Int
    mortars::Mortars
    mortar_id::Int
    boundaries::Boundaries
    boundary_id::Int
    mpi_interfaces::MPIInterfaces
    mpi_interface_id::Int
    mpi_mortars::MPIMortars
    mpi_mortar_id::Int
    mesh::Mesh
end

function ParallelInitSurfacesIterFaceUserData(interfaces, mortars, boundaries,
                                              mpi_interfaces, mpi_mortars, mesh)
    return ParallelInitSurfacesIterFaceUserData{typeof(interfaces), typeof(mortars),
                                                typeof(boundaries),
                                                typeof(mpi_interfaces),
                                                typeof(mpi_mortars), typeof(mesh)}(interfaces,
                                                                                   1,
                                                                                   mortars,
                                                                                   1,
                                                                                   boundaries,
                                                                                   1,
                                                                                   mpi_interfaces,
                                                                                   1,
                                                                                   mpi_mortars,
                                                                                   1,
                                                                                   mesh)
end

function init_surfaces_iter_face_parallel(info, user_data)
    # Unpack user_data
    data = unsafe_pointer_to_objref(Ptr{ParallelInitSurfacesIterFaceUserData}(user_data))

    # Function barrier because the unpacked user_data above is type-unstable
    init_surfaces_iter_face_inner(info, data)
end

# 2D
function cfunction(::typeof(init_surfaces_iter_face_parallel), ::Val{2})
    @cfunction(init_surfaces_iter_face_parallel, Cvoid,
               (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(init_surfaces_iter_face_parallel), ::Val{3})
    @cfunction(init_surfaces_iter_face_parallel, Cvoid,
               (Ptr{p8est_iter_face_info_t}, Ptr{Cvoid}))
end

# Function barrier for type stability, overload for parallel P4estMesh
function init_surfaces_iter_face_inner(info,
                                       user_data::ParallelInitSurfacesIterFaceUserData)
    @unpack interfaces, mortars, boundaries, mpi_interfaces, mpi_mortars = user_data
    # This function is called during `init_surfaces!`, more precisely it is called for each face
    # while p4est iterates over the forest. Since `init_surfaces!` can be used to initialize all
    # surfaces at once or any subset of them, some of the unpacked values above may be `nothing` if
    # they're not supposed to be initialized during this call. That is why we need additional
    # `!== nothing` checks below before initializing individual faces.
    info_pw = PointerWrapper(info)
    if info_pw.sides.elem_count[] == 2
        # Two neighboring elements => Interface or mortar

        # Extract surface data
        sides_pw = (load_pointerwrapper_side(info_pw, 1),
                    load_pointerwrapper_side(info_pw, 2))

        if sides_pw[1].is_hanging[] == false && sides_pw[2].is_hanging[] == false
            # No hanging nodes => normal interface or MPI interface
            if sides_pw[1].is.full.is_ghost[] == true ||
               sides_pw[2].is.full.is_ghost[] == true # remote side => MPI interface
                if mpi_interfaces !== nothing
                    init_mpi_interfaces_iter_face_inner(info_pw, sides_pw, user_data)
                end
            else
                if interfaces !== nothing
                    init_interfaces_iter_face_inner(info_pw, sides_pw, user_data)
                end
            end
        else
            # Hanging nodes => mortar or MPI mortar
            # First, we check which side is hanging, i.e., on which side we have the refined cells.
            # Then we check if any of the refined cells or the coarse cell are "ghost" cells, i.e., they
            # belong to another rank. That way we can determine if this is a regular mortar or MPI mortar
            if sides_pw[1].is_hanging[] == true
                @assert sides_pw[2].is_hanging[] == false
                if any(sides_pw[1].is.hanging.is_ghost[] .== true) ||
                   sides_pw[2].is.full.is_ghost[] == true
                    face_has_ghost_side = true
                else
                    face_has_ghost_side = false
                end
            else # sides_pw[2].is_hanging[] == true
                @assert sides_pw[1].is_hanging[] == false
                if sides_pw[1].is.full.is_ghost[] == true ||
                   any(sides_pw[2].is.hanging.is_ghost[] .== true)
                    face_has_ghost_side = true
                else
                    face_has_ghost_side = false
                end
            end
            # Initialize mortar or MPI mortar
            if face_has_ghost_side && mpi_mortars !== nothing
                init_mpi_mortars_iter_face_inner(info_pw, sides_pw, user_data)
            elseif !face_has_ghost_side && mortars !== nothing
                init_mortars_iter_face_inner(info_pw, sides_pw, user_data)
            end
        end
    elseif info_pw.sides.elem_count[] == 1
        # One neighboring elements => boundary
        if boundaries !== nothing
            init_boundaries_iter_face_inner(info_pw, user_data)
        end
    end

    return nothing
end

function init_surfaces!(interfaces, mortars, boundaries, mpi_interfaces, mpi_mortars,
                        mesh::ParallelP4estMesh)
    # Let p4est iterate over all interfaces and call init_surfaces_iter_face
    iter_face_c = cfunction(init_surfaces_iter_face_parallel, Val(ndims(mesh)))
    user_data = ParallelInitSurfacesIterFaceUserData(interfaces, mortars, boundaries,
                                                     mpi_interfaces, mpi_mortars, mesh)

    iterate_p4est(mesh.p4est, user_data; ghost_layer = mesh.ghost,
                  iter_face_c = iter_face_c)

    return nothing
end

# Initialization of MPI interfaces after the function barrier
function init_mpi_interfaces_iter_face_inner(info_pw, sides_pw, user_data)
    @unpack mpi_interfaces, mpi_interface_id, mesh = user_data
    user_data.mpi_interface_id += 1

    if sides_pw[1].is.full.is_ghost[] == true
        local_side = 2
    elseif sides_pw[2].is.full.is_ghost[] == true
        local_side = 1
    else
        error("should not happen")
    end

    # Get local tree, one-based indexing
    tree_pw = load_pointerwrapper_tree(mesh.p4est, sides_pw[local_side].treeid[] + 1)
    # Quadrant numbering offset of the local quadrant at this interface
    offset = tree_pw.quadrants_offset[]
    tree_quad_id = sides_pw[local_side].is.full.quadid[] # quadid in the local tree
    # ID of the local neighboring quad, cumulative over local trees
    local_quad_id = offset + tree_quad_id

    # p4est uses zero-based indexing, convert to one-based indexing
    mpi_interfaces.local_neighbor_ids[mpi_interface_id] = local_quad_id + 1
    mpi_interfaces.local_sides[mpi_interface_id] = local_side

    # Face at which the interface lies
    faces = (sides_pw[1].face[], sides_pw[2].face[])

    # Save mpi_interfaces.node_indices dimension specific in containers_[23]d_parallel.jl
    init_mpi_interface_node_indices!(mpi_interfaces, faces, local_side,
                                     info_pw.orientation[],
                                     mpi_interface_id)

    return nothing
end

# Initialization of MPI mortars after the function barrier
function init_mpi_mortars_iter_face_inner(info_pw, sides_pw, user_data)
    @unpack mpi_mortars, mpi_mortar_id, mesh = user_data
    user_data.mpi_mortar_id += 1

    # Get Tuple of adjacent trees, one-based indexing
    trees_pw = (load_pointerwrapper_tree(mesh.p4est, sides_pw[1].treeid[] + 1),
                load_pointerwrapper_tree(mesh.p4est, sides_pw[2].treeid[] + 1))
    # Quadrant numbering offsets of the quadrants at this mortar
    offsets = SVector(trees_pw[1].quadrants_offset[],
                      trees_pw[2].quadrants_offset[])

    if sides_pw[1].is_hanging[] == true
        hanging_side = 1
        full_side = 2
    else # sides_pw[2].is_hanging[] == true
        hanging_side = 2
        full_side = 1
    end
    # Just be sure before accessing is.full or is.hanging later
    @assert sides_pw[full_side].is_hanging[] == false
    @assert sides_pw[hanging_side].is_hanging[] == true

    # Find small quads that are locally available
    local_small_quad_positions = findall(sides_pw[hanging_side].is.hanging.is_ghost[] .==
                                         false)

    # Get id of local small quadrants within their tree
    # Indexing CBinding.Caccessor via a Vector does not work here -> use map instead
    tree_small_quad_ids = map(p -> sides_pw[hanging_side].is.hanging.quadid[][p],
                              local_small_quad_positions)
    local_small_quad_ids = offsets[hanging_side] .+ tree_small_quad_ids # ids cumulative over local trees

    # Determine if large quadrant is available and if yes, determine its id
    if sides_pw[full_side].is.full.is_ghost[] == false
        local_large_quad_id = offsets[full_side] + sides_pw[full_side].is.full.quadid[]
    else
        local_large_quad_id = -1 # large quad is ghost
    end

    # Write data to mortar container, convert to 1-based indexing
    # Start with small elements
    local_neighbor_ids = local_small_quad_ids .+ 1
    local_neighbor_positions = local_small_quad_positions
    # Add large element information if it is locally available
    if local_large_quad_id > -1
        push!(local_neighbor_ids, local_large_quad_id + 1) # convert to 1-based index
        push!(local_neighbor_positions, 2^(ndims(mesh) - 1) + 1)
    end

    mpi_mortars.local_neighbor_ids[mpi_mortar_id] = local_neighbor_ids
    mpi_mortars.local_neighbor_positions[mpi_mortar_id] = local_neighbor_positions

    # init_mortar_node_indices! expects side 1 to contain small elements
    faces = (sides_pw[hanging_side].face[], sides_pw[full_side].face[])
    init_mortar_node_indices!(mpi_mortars, faces, info_pw.orientation[], mpi_mortar_id)

    return nothing
end

# Iterate over all interfaces and count
# - (inner) interfaces
# - mortars
# - boundaries
# - (MPI) interfaces at subdomain boundaries
# - (MPI) mortars at subdomain boundaries
# and collect the numbers in `user_data` in this order.
function count_surfaces_iter_face_parallel(info, user_data)
    info_pw = PointerWrapper(info)
    if info_pw.sides.elem_count[] == 2
        # Two neighboring elements => Interface or mortar

        # Extract surface data
        sides_pw = (load_pointerwrapper_side(info_pw, 1),
                    load_pointerwrapper_side(info_pw, 2))

        if sides_pw[1].is_hanging[] == false && sides_pw[2].is_hanging[] == false
            # No hanging nodes => normal interface or MPI interface
            if sides_pw[1].is.full.is_ghost[] == true ||
               sides_pw[2].is.full.is_ghost[] == true # remote side => MPI interface
                # Unpack user_data = [mpi_interface_count] and increment mpi_interface_count
                pw = PointerWrapper(Int, user_data)
                id = pw[4]
                pw[4] = id + 1
            else
                # Unpack user_data = [interface_count] and increment interface_count
                pw = PointerWrapper(Int, user_data)
                id = pw[1]
                pw[1] = id + 1
            end
        else
            # Hanging nodes => mortar or MPI mortar
            # First, we check which side is hanging, i.e., on which side we have the refined cells.
            # Then we check if any of the refined cells or the coarse cell are "ghost" cells, i.e., they
            # belong to another rank. That way we can determine if this is a regular mortar or MPI mortar
            if sides_pw[1].is_hanging[] == true
                @assert sides_pw[2].is_hanging[] == false
                if any(sides_pw[1].is.hanging.is_ghost[] .== true) ||
                   sides_pw[2].is.full.is_ghost[] == true
                    face_has_ghost_side = true
                else
                    face_has_ghost_side = false
                end
            else # sides_pw[2].is_hanging[] == true
                @assert sides_pw[1].is_hanging[] == false
                if sides_pw[1].is.full.is_ghost[] == true ||
                   any(sides_pw[2].is.hanging.is_ghost[] .== true)
                    face_has_ghost_side = true
                else
                    face_has_ghost_side = false
                end
            end
            if face_has_ghost_side
                # Unpack user_data = [mpi_mortar_count] and increment mpi_mortar_count
                pw = PointerWrapper(Int, user_data)
                id = pw[5]
                pw[5] = id + 1
            else
                # Unpack user_data = [mortar_count] and increment mortar_count
                pw = PointerWrapper(Int, user_data)
                id = pw[2]
                pw[2] = id + 1
            end
        end
    elseif info_pw.sides.elem_count[] == 1
        # One neighboring elements => boundary

        # Unpack user_data = [boundary_count] and increment boundary_count
        pw = PointerWrapper(Int, user_data)
        id = pw[3]
        pw[3] = id + 1
    end

    return nothing
end

# 2D
function cfunction(::typeof(count_surfaces_iter_face_parallel), ::Val{2})
    @cfunction(count_surfaces_iter_face_parallel, Cvoid,
               (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(count_surfaces_iter_face_parallel), ::Val{3})
    @cfunction(count_surfaces_iter_face_parallel, Cvoid,
               (Ptr{p8est_iter_face_info_t}, Ptr{Cvoid}))
end

function count_required_surfaces(mesh::ParallelP4estMesh)
    # Let p4est iterate over all interfaces and call count_surfaces_iter_face_parallel
    iter_face_c = cfunction(count_surfaces_iter_face_parallel, Val(ndims(mesh)))

    # interfaces, mortars, boundaries, mpi_interfaces, mpi_mortars
    user_data = [0, 0, 0, 0, 0]

    iterate_p4est(mesh.p4est, user_data; ghost_layer = mesh.ghost,
                  iter_face_c = iter_face_c)

    # Return counters
    return (interfaces = user_data[1],
            mortars = user_data[2],
            boundaries = user_data[3],
            mpi_interfaces = user_data[4],
            mpi_mortars = user_data[5])
end
end # @muladd
