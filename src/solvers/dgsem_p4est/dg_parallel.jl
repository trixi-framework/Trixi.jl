# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct P4estMPICache{uEltype}
    mpi_neighbor_ranks::Vector{Int}
    mpi_neighbor_interfaces::Vector{Vector{Int}}
    mpi_neighbor_mortars::Vector{Vector{Int}}
    mpi_send_buffers::Vector{Vector{uEltype}}
    mpi_recv_buffers::Vector{Vector{uEltype}}
    mpi_send_requests::Vector{MPI.Request}
    mpi_recv_requests::Vector{MPI.Request}
    n_elements_by_rank::OffsetArray{Int, 1, Array{Int, 1}}
    n_elements_global::Int
    first_element_global_id::Int
end

function P4estMPICache(uEltype)
    # MPI communication "just works" for bitstypes only
    if !isbitstype(uEltype)
        throw(ArgumentError("P4estMPICache only supports bitstypes, $uEltype is not a bitstype."))
    end

    mpi_neighbor_ranks = Vector{Int}(undef, 0)
    mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, 0)
    mpi_neighbor_mortars = Vector{Vector{Int}}(undef, 0)
    mpi_send_buffers = Vector{Vector{uEltype}}(undef, 0)
    mpi_recv_buffers = Vector{Vector{uEltype}}(undef, 0)
    mpi_send_requests = Vector{MPI.Request}(undef, 0)
    mpi_recv_requests = Vector{MPI.Request}(undef, 0)
    n_elements_by_rank = OffsetArray(Vector{Int}(undef, 0), 0:-1)
    n_elements_global = 0
    first_element_global_id = 0

    P4estMPICache{uEltype}(mpi_neighbor_ranks, mpi_neighbor_interfaces,
                           mpi_neighbor_mortars,
                           mpi_send_buffers, mpi_recv_buffers,
                           mpi_send_requests, mpi_recv_requests,
                           n_elements_by_rank, n_elements_global,
                           first_element_global_id)
end

@inline Base.eltype(::P4estMPICache{uEltype}) where {uEltype} = uEltype

function start_mpi_send!(mpi_cache::P4estMPICache, mesh, equations, dg, cache)
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)
    n_small_elements = 2^(ndims(mesh) - 1)

    for rank in 1:length(mpi_cache.mpi_neighbor_ranks)
        send_buffer = mpi_cache.mpi_send_buffers[rank]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[rank])
            first = (index - 1) * data_size + 1
            last = (index - 1) * data_size + data_size
            local_side = cache.mpi_interfaces.local_sides[interface]
            @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[local_side, ..,
                                                                         interface])
        end

        # Set send_buffer corresponding to mortar data to NaN and overwrite the parts where local
        # data exists
        interfaces_data_size = length(mpi_cache.mpi_neighbor_interfaces[rank]) *
                               data_size
        mortars_data_size = length(mpi_cache.mpi_neighbor_mortars[rank]) *
                            n_small_elements * 2 * data_size
        # `NaN |> eltype(...)` ensures that the NaN's are of the appropriate floating point type
        send_buffer[(interfaces_data_size + 1):(interfaces_data_size + mortars_data_size)] .= NaN |>
                                                                                              eltype(mpi_cache)

        for (index, mortar) in enumerate(mpi_cache.mpi_neighbor_mortars[rank])
            index_base = interfaces_data_size +
                         (index - 1) * n_small_elements * 2 * data_size
            indices = buffer_mortar_indices(mesh, index_base, data_size)

            for position in cache.mpi_mortars.local_neighbor_positions[mortar]
                first, last = indices[position]
                if position > n_small_elements # large element
                    @views send_buffer[first:last] .= vec(cache.mpi_mortars.u[2, :, :,
                                                                              ..,
                                                                              mortar])
                else # small element
                    @views send_buffer[first:last] .= vec(cache.mpi_mortars.u[1, :,
                                                                              position,
                                                                              ..,
                                                                              mortar])
                end
            end
        end
    end

    # Start sending
    for (index, rank) in enumerate(mpi_cache.mpi_neighbor_ranks)
        mpi_cache.mpi_send_requests[index] = MPI.Isend(mpi_cache.mpi_send_buffers[index],
                                                       rank, mpi_rank(), mpi_comm())
    end

    return nothing
end

function start_mpi_receive!(mpi_cache::P4estMPICache)
    for (index, rank) in enumerate(mpi_cache.mpi_neighbor_ranks)
        mpi_cache.mpi_recv_requests[index] = MPI.Irecv!(mpi_cache.mpi_recv_buffers[index],
                                                        rank, rank, mpi_comm())
    end

    return nothing
end

function finish_mpi_send!(mpi_cache::P4estMPICache)
    MPI.Waitall(mpi_cache.mpi_send_requests, MPI.Status)
end

function finish_mpi_receive!(mpi_cache::P4estMPICache, mesh, equations, dg, cache)
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)
    n_small_elements = 2^(ndims(mesh) - 1)
    n_positions = n_small_elements + 1

    # Start receiving and unpack received data until all communication is finished
    data = MPI.Waitany(mpi_cache.mpi_recv_requests)
    while data !== nothing
        recv_buffer = mpi_cache.mpi_recv_buffers[data]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[data])
            first = (index - 1) * data_size + 1
            last = (index - 1) * data_size + data_size

            if cache.mpi_interfaces.local_sides[interface] == 1 # local element on primary side
                @views vec(cache.mpi_interfaces.u[2, .., interface]) .= recv_buffer[first:last]
            else # local element at secondary side
                @views vec(cache.mpi_interfaces.u[1, .., interface]) .= recv_buffer[first:last]
            end
        end

        interfaces_data_size = length(mpi_cache.mpi_neighbor_interfaces[data]) *
                               data_size
        for (index, mortar) in enumerate(mpi_cache.mpi_neighbor_mortars[data])
            index_base = interfaces_data_size +
                         (index - 1) * n_small_elements * 2 * data_size
            indices = buffer_mortar_indices(mesh, index_base, data_size)

            for position in 1:n_positions
                # Skip if received data for `position` is NaN as no real data has been sent for the
                # corresponding element
                if isnan(recv_buffer[Base.first(indices[position])])
                    continue
                end

                first, last = indices[position]
                if position == n_positions # large element
                    @views vec(cache.mpi_mortars.u[2, :, :, .., mortar]) .= recv_buffer[first:last]
                else # small element
                    @views vec(cache.mpi_mortars.u[1, :, position, .., mortar]) .= recv_buffer[first:last]
                end
            end
        end

        data = MPI.Waitany(mpi_cache.mpi_recv_requests)
    end

    return nothing
end

# Return a tuple `indices` where indices[position] is a `(first, last)` tuple for accessing the
# data corresponding to the `position` part of a mortar in an MPI buffer. The mortar data must begin
# at `index_base`+1 in the MPI buffer. `data_size` is the data size associated with each small
# position (i.e. position 1 or 2). The data corresponding to the large side (i.e. position 3) has
# size `2 * data_size`.
@inline function buffer_mortar_indices(mesh::Union{ParallelP4estMesh{2},
                                                   ParallelT8codeMesh{2}}, index_base,
                                       data_size)
    return (
            # first, last for local element in position 1 (small element)
            (index_base + 1,
             index_base + 1 * data_size),
            # first, last for local element in position 2 (small element)
            (index_base + 1 * data_size + 1,
             index_base + 2 * data_size),
            # first, last for local element in position 3 (large element)
            (index_base + 2 * data_size + 1,
             index_base + 4 * data_size))
end

# Return a tuple `indices` where indices[position] is a `(first, last)` tuple for accessing the
# data corresponding to the `position` part of a mortar in an MPI buffer. The mortar data must begin
# at `index_base`+1 in the MPI buffer. `data_size` is the data size associated with each small
# position (i.e. position 1 to 4). The data corresponding to the large side (i.e. position 5) has
# size `4 * data_size`.
@inline function buffer_mortar_indices(mesh::Union{ParallelP4estMesh{3},
                                                   ParallelT8codeMesh{3}}, index_base,
                                       data_size)
    return (
            # first, last for local element in position 1 (small element)
            (index_base + 1,
             index_base + 1 * data_size),
            # first, last for local element in position 2 (small element)
            (index_base + 1 * data_size + 1,
             index_base + 2 * data_size),
            # first, last for local element in position 3 (small element)
            (index_base + 2 * data_size + 1,
             index_base + 3 * data_size),
            # first, last for local element in position 4 (small element)
            (index_base + 3 * data_size + 1,
             index_base + 4 * data_size),
            # first, last for local element in position 5 (large element)
            (index_base + 4 * data_size + 1,
             index_base + 8 * data_size))
end

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::ParallelP4estMesh, equations::AbstractEquations, dg::DG,
                      ::Any, ::Type{uEltype}) where {uEltype <: Real}
    # Make sure to balance and partition the p4est and create a new ghost layer before creating any
    # containers in case someone has tampered with the p4est after creating the mesh
    balance!(mesh)
    partition!(mesh)
    update_ghost_layer!(mesh)

    elements = init_elements(mesh, equations, dg.basis, uEltype)

    mpi_interfaces = init_mpi_interfaces(mesh, equations, dg.basis, elements)
    mpi_mortars = init_mpi_mortars(mesh, equations, dg.basis, elements)
    mpi_cache = init_mpi_cache(mesh, mpi_interfaces, mpi_mortars,
                               nvariables(equations), nnodes(dg), uEltype)

    exchange_normal_directions!(mpi_mortars, mpi_cache, mesh, nnodes(dg))

    interfaces = init_interfaces(mesh, equations, dg.basis, elements)
    boundaries = init_boundaries(mesh, equations, dg.basis, elements)
    mortars = init_mortars(mesh, equations, dg.basis, elements)

    cache = (; elements, interfaces, mpi_interfaces, boundaries, mortars, mpi_mortars,
             mpi_cache)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
    cache = (; cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

    return cache
end

function init_mpi_cache(mesh::ParallelP4estMesh, mpi_interfaces, mpi_mortars, nvars,
                        nnodes, uEltype)
    mpi_cache = P4estMPICache(uEltype)
    init_mpi_cache!(mpi_cache, mesh, mpi_interfaces, mpi_mortars, nvars, nnodes,
                    uEltype)

    return mpi_cache
end

function init_mpi_cache!(mpi_cache::P4estMPICache, mesh::ParallelP4estMesh,
                         mpi_interfaces, mpi_mortars, nvars, n_nodes, uEltype)
    mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars = init_mpi_neighbor_connectivity(mpi_interfaces,
                                                                                                       mpi_mortars,
                                                                                                       mesh)

    mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests = init_mpi_data_structures(mpi_neighbor_interfaces,
                                                                                                        mpi_neighbor_mortars,
                                                                                                        ndims(mesh),
                                                                                                        nvars,
                                                                                                        n_nodes,
                                                                                                        uEltype)

    # Determine local and total number of elements
    n_elements_global = Int(mesh.p4est.global_num_quadrants[])
    n_elements_by_rank = vcat(Int.(unsafe_wrap(Array, mesh.p4est.global_first_quadrant,
                                               mpi_nranks())),
                              n_elements_global) |> diff # diff sufficient due to 0-based quad indices
    n_elements_by_rank = OffsetArray(n_elements_by_rank, 0:(mpi_nranks() - 1))
    # Account for 1-based indexing in Julia
    first_element_global_id = Int(mesh.p4est.global_first_quadrant[mpi_rank() + 1]) + 1
    @assert n_elements_global==sum(n_elements_by_rank) "error in total number of elements"

    # TODO reuse existing structures
    @pack! mpi_cache = mpi_neighbor_ranks, mpi_neighbor_interfaces,
                       mpi_neighbor_mortars,
                       mpi_send_buffers, mpi_recv_buffers,
                       mpi_send_requests, mpi_recv_requests,
                       n_elements_by_rank, n_elements_global,
                       first_element_global_id
end

function init_mpi_neighbor_connectivity(mpi_interfaces, mpi_mortars,
                                        mesh::ParallelP4estMesh)
    # Let p4est iterate over all interfaces and call init_neighbor_rank_connectivity_iter_face
    # to collect connectivity information
    iter_face_c = cfunction(init_neighbor_rank_connectivity_iter_face, Val(ndims(mesh)))
    user_data = InitNeighborRankConnectivityIterFaceUserData(mpi_interfaces,
                                                             mpi_mortars, mesh)

    iterate_p4est(mesh.p4est, user_data; ghost_layer = mesh.ghost,
                  iter_face_c = iter_face_c)

    # Build proper connectivity data structures from information gathered by iterating over p4est
    @unpack global_interface_ids, neighbor_ranks_interface, global_mortar_ids, neighbor_ranks_mortar = user_data

    mpi_neighbor_ranks = vcat(neighbor_ranks_interface, neighbor_ranks_mortar...) |>
                         sort |> unique

    p = sortperm(global_interface_ids)
    neighbor_ranks_interface .= neighbor_ranks_interface[p]
    interface_ids = collect(1:nmpiinterfaces(mpi_interfaces))[p]

    p = sortperm(global_mortar_ids)
    neighbor_ranks_mortar .= neighbor_ranks_mortar[p]
    mortar_ids = collect(1:nmpimortars(mpi_mortars))[p]

    # For each neighbor rank, init connectivity data structures
    mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
    mpi_neighbor_mortars = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
    for (index, rank) in enumerate(mpi_neighbor_ranks)
        mpi_neighbor_interfaces[index] = interface_ids[findall(==(rank),
                                                               neighbor_ranks_interface)]
        mpi_neighbor_mortars[index] = mortar_ids[findall(x -> (rank in x),
                                                         neighbor_ranks_mortar)]
    end

    # Check that all interfaces were counted exactly once
    @assert mapreduce(length, +, mpi_neighbor_interfaces; init = 0) ==
            nmpiinterfaces(mpi_interfaces)

    return mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars
end

mutable struct InitNeighborRankConnectivityIterFaceUserData{MPIInterfaces, MPIMortars,
                                                            Mesh}
    interfaces::MPIInterfaces
    interface_id::Int
    global_interface_ids::Vector{Int}
    neighbor_ranks_interface::Vector{Int}
    mortars::MPIMortars
    mortar_id::Int
    global_mortar_ids::Vector{Int}
    neighbor_ranks_mortar::Vector{Vector{Int}}
    mesh::Mesh
end

function InitNeighborRankConnectivityIterFaceUserData(mpi_interfaces, mpi_mortars, mesh)
    global_interface_ids = fill(-1, nmpiinterfaces(mpi_interfaces))
    neighbor_ranks_interface = fill(-1, nmpiinterfaces(mpi_interfaces))
    global_mortar_ids = fill(-1, nmpimortars(mpi_mortars))
    neighbor_ranks_mortar = Vector{Vector{Int}}(undef, nmpimortars(mpi_mortars))

    return InitNeighborRankConnectivityIterFaceUserData{typeof(mpi_interfaces),
                                                        typeof(mpi_mortars),
                                                        typeof(mesh)}(mpi_interfaces, 1,
                                                                      global_interface_ids,
                                                                      neighbor_ranks_interface,
                                                                      mpi_mortars, 1,
                                                                      global_mortar_ids,
                                                                      neighbor_ranks_mortar,
                                                                      mesh)
end

function init_neighbor_rank_connectivity_iter_face(info, user_data)
    data = unsafe_pointer_to_objref(Ptr{InitNeighborRankConnectivityIterFaceUserData}(user_data))

    # Function barrier because the unpacked user_data above is not type-stable
    init_neighbor_rank_connectivity_iter_face_inner(info, data)
end

# 2D
function cfunction(::typeof(init_neighbor_rank_connectivity_iter_face), ::Val{2})
    @cfunction(init_neighbor_rank_connectivity_iter_face, Cvoid,
               (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
end
# 3D
function cfunction(::typeof(init_neighbor_rank_connectivity_iter_face), ::Val{3})
    @cfunction(init_neighbor_rank_connectivity_iter_face, Cvoid,
               (Ptr{p8est_iter_face_info_t}, Ptr{Cvoid}))
end

# Function barrier for type stability
function init_neighbor_rank_connectivity_iter_face_inner(info, user_data)
    @unpack interfaces, interface_id, global_interface_ids, neighbor_ranks_interface,
    mortars, mortar_id, global_mortar_ids, neighbor_ranks_mortar, mesh = user_data

    info_pw = PointerWrapper(info)
    # Get the global interface/mortar ids and neighbor rank if current face belongs to an MPI
    # interface/mortar
    if info_pw.sides.elem_count[] == 2 # MPI interfaces/mortars have two neighboring elements
        # Extract surface data
        sides_pw = (load_pointerwrapper_side(info_pw, 1),
                    load_pointerwrapper_side(info_pw, 2))

        if sides_pw[1].is_hanging[] == false && sides_pw[2].is_hanging[] == false # No hanging nodes for MPI interfaces
            if sides_pw[1].is.full.is_ghost[] == true
                remote_side = 1
                local_side = 2
            elseif sides_pw[2].is.full.is_ghost[] == true
                remote_side = 2
                local_side = 1
            else # both sides are on this rank -> skip since it's a regular interface
                return nothing
            end

            # Sanity check, current face should belong to current MPI interface
            local_tree_pw = load_pointerwrapper_tree(mesh.p4est,
                                                     sides_pw[local_side].treeid[] + 1) # one-based indexing
            local_quad_id = local_tree_pw.quadrants_offset[] +
                            sides_pw[local_side].is.full.quadid[]
            @assert interfaces.local_neighbor_ids[interface_id] == local_quad_id + 1 # one-based indexing

            # Get neighbor ID from ghost layer
            proc_offsets = unsafe_wrap(Array,
                                       info_pw.ghost_layer.proc_offsets,
                                       mpi_nranks() + 1)
            ghost_id = sides_pw[remote_side].is.full.quadid[] # indexes the ghost layer, 0-based
            neighbor_rank = findfirst(r -> proc_offsets[r] <= ghost_id <
                                           proc_offsets[r + 1],
                                      1:mpi_nranks()) - 1 # MPI ranks are 0-based
            neighbor_ranks_interface[interface_id] = neighbor_rank

            # Global interface id is the globally unique quadrant id of the quadrant on the primary
            # side (1) multiplied by the number of faces per quadrant plus face
            if local_side == 1
                offset = mesh.p4est.global_first_quadrant[mpi_rank() + 1] # one-based indexing
                primary_quad_id = offset + local_quad_id
            else
                offset = mesh.p4est.global_first_quadrant[neighbor_rank + 1] # one-based indexing
                primary_quad_id = offset + sides_pw[1].is.full.quad.p.piggy3.local_num[]
            end
            global_interface_id = 2 * ndims(mesh) * primary_quad_id + sides_pw[1].face[]
            global_interface_ids[interface_id] = global_interface_id

            user_data.interface_id += 1
        else # hanging node
            if sides_pw[1].is_hanging[] == true
                hanging_side = 1
                full_side = 2
            else
                hanging_side = 2
                full_side = 1
            end
            # Verify before accessing is.full / is.hanging
            @assert sides_pw[hanging_side].is_hanging[] == true &&
                    sides_pw[full_side].is_hanging[] == false

            # If all quadrants are locally available, this is a regular mortar -> skip
            if sides_pw[full_side].is.full.is_ghost[] == false &&
               all(sides_pw[hanging_side].is.hanging.is_ghost[] .== false)
                return nothing
            end

            trees_pw = (load_pointerwrapper_tree(mesh.p4est, sides_pw[1].treeid[] + 1),
                        load_pointerwrapper_tree(mesh.p4est, sides_pw[2].treeid[] + 1))

            # Find small quads that are remote and determine which rank owns them
            remote_small_quad_positions = findall(sides_pw[hanging_side].is.hanging.is_ghost[] .==
                                                  true)
            proc_offsets = unsafe_wrap(Array,
                                       info_pw.ghost_layer.proc_offsets,
                                       mpi_nranks() + 1)
            # indices of small remote quads inside the ghost layer, 0-based
            ghost_ids = map(pos -> sides_pw[hanging_side].is.hanging.quadid[][pos],
                            remote_small_quad_positions)
            neighbor_ranks = map(ghost_ids) do ghost_id
                return findfirst(r -> proc_offsets[r] <= ghost_id < proc_offsets[r + 1],
                                 1:mpi_nranks()) - 1 # MPI ranks are 0-based
            end
            # Determine global quad id of large element to determine global MPI mortar id
            # Furthermore, if large element is ghost, add its owner rank to neighbor_ranks
            if sides_pw[full_side].is.full.is_ghost[] == true
                ghost_id = sides_pw[full_side].is.full.quadid[]
                large_quad_owner_rank = findfirst(r -> proc_offsets[r] <= ghost_id <
                                                       proc_offsets[r + 1],
                                                  1:mpi_nranks()) - 1 # MPI ranks are 0-based
                push!(neighbor_ranks, large_quad_owner_rank)

                offset = mesh.p4est.global_first_quadrant[large_quad_owner_rank + 1] # one-based indexing
                large_quad_id = offset +
                                sides_pw[full_side].is.full.quad.p.piggy3.local_num[]
            else
                offset = mesh.p4est.global_first_quadrant[mpi_rank() + 1] # one-based indexing
                large_quad_id = offset + trees_pw[full_side].quadrants_offset[] +
                                sides_pw[full_side].is.full.quadid[]
            end
            neighbor_ranks_mortar[mortar_id] = neighbor_ranks
            # Global mortar id is the globally unique quadrant id of the large quadrant multiplied by the
            # number of faces per quadrant plus face
            global_mortar_ids[mortar_id] = 2 * ndims(mesh) * large_quad_id +
                                           sides_pw[full_side].face[]

            user_data.mortar_id += 1
        end
    end

    return nothing
end

# Exchange normal directions of small elements of the MPI mortars. They are needed on all involved
# MPI ranks to calculate the mortar fluxes.
function exchange_normal_directions!(mpi_mortars, mpi_cache,
                                     mesh::Union{ParallelP4estMesh, ParallelT8codeMesh},
                                     n_nodes)
    RealT = real(mesh)
    n_dims = ndims(mesh)
    @unpack mpi_neighbor_mortars, mpi_neighbor_ranks = mpi_cache
    n_small_elements = 2^(n_dims - 1)
    data_size = n_nodes^(n_dims - 1) * n_dims

    # Create buffers and requests
    send_buffers = Vector{Vector{RealT}}(undef, length(mpi_neighbor_mortars))
    recv_buffers = Vector{Vector{RealT}}(undef, length(mpi_neighbor_mortars))
    for index in 1:length(mpi_neighbor_mortars)
        send_buffers[index] = Vector{RealT}(undef,
                                            length(mpi_neighbor_mortars[index]) *
                                            n_small_elements * data_size)
        send_buffers[index] .= NaN |> RealT
        recv_buffers[index] = Vector{RealT}(undef,
                                            length(mpi_neighbor_mortars[index]) *
                                            n_small_elements * data_size)
        recv_buffers[index] .= NaN |> RealT
    end
    send_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_mortars))
    recv_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_mortars))

    # Fill send buffers
    for rank in 1:length(mpi_neighbor_ranks)
        send_buffer = send_buffers[rank]

        for (index, mortar) in enumerate(mpi_neighbor_mortars[rank])
            index_base = (index - 1) * n_small_elements * data_size
            indices = buffer_mortar_indices(mesh, index_base, data_size)
            for position in mpi_mortars.local_neighbor_positions[mortar]
                if position <= n_small_elements # element is small
                    first, last = indices[position]
                    @views send_buffer[first:last] .= vec(mpi_mortars.normal_directions[:,
                                                                                        ..,
                                                                                        position,
                                                                                        mortar])
                end
            end
        end
    end

    # Start data exchange
    for (index, rank) in enumerate(mpi_neighbor_ranks)
        send_requests[index] = MPI.Isend(send_buffers[index], rank, mpi_rank(),
                                         mpi_comm())
        recv_requests[index] = MPI.Irecv!(recv_buffers[index], rank, rank, mpi_comm())
    end

    # Unpack data from receive buffers
    data = MPI.Waitany(recv_requests)
    while data !== nothing
        recv_buffer = recv_buffers[data]

        for (index, mortar) in enumerate(mpi_neighbor_mortars[data])
            index_base = (index - 1) * n_small_elements * data_size
            indices = buffer_mortar_indices(mesh, index_base, data_size)
            for position in 1:n_small_elements
                # Skip if received data for `position` is NaN as no real data has been sent for the
                # corresponding element
                if isnan(recv_buffer[Base.first(indices[position])])
                    continue
                end

                first, last = indices[position]
                @views vec(mpi_mortars.normal_directions[:, .., position, mortar]) .= recv_buffer[first:last]
            end
        end

        data = MPI.Waitany(recv_requests)
    end

    # Wait for communication to finish
    MPI.Waitall(send_requests, MPI.Status)

    return nothing
end

# Get normal direction of MPI mortar
@inline function get_normal_direction(mpi_mortars::P4estMPIMortarContainer, indices...)
    SVector(ntuple(@inline(dim->mpi_mortars.normal_directions[dim, indices...]),
                   Val(ndims(mpi_mortars))))
end

include("dg_2d_parallel.jl")
include("dg_3d_parallel.jl")
end # muladd
