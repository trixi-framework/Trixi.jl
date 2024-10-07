# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# everything related to a DG semidiscretization in 2D using MPI,
# currently limited to Lobatto-Legendre nodes

# TODO: MPI dimension agnostic
mutable struct MPICache{uEltype <: Real}
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

function MPICache(uEltype)
    # MPI communication "just works" for bitstypes only
    if !isbitstype(uEltype)
        throw(ArgumentError("MPICache only supports bitstypes, $uEltype is not a bitstype."))
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

    MPICache{uEltype}(mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars,
                      mpi_send_buffers, mpi_recv_buffers,
                      mpi_send_requests, mpi_recv_requests,
                      n_elements_by_rank, n_elements_global,
                      first_element_global_id)
end
@inline Base.eltype(::MPICache{uEltype}) where {uEltype} = uEltype

# TODO: MPI dimension agnostic
function start_mpi_receive!(mpi_cache::MPICache)
    for (index, rank) in enumerate(mpi_cache.mpi_neighbor_ranks)
        mpi_cache.mpi_recv_requests[index] = MPI.Irecv!(mpi_cache.mpi_recv_buffers[index],
                                                        rank, rank, mpi_comm())
    end

    return nothing
end

# TODO: MPI dimension agnostic
function start_mpi_send!(mpi_cache::MPICache, mesh, equations, dg, cache)
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    for rank in 1:length(mpi_cache.mpi_neighbor_ranks)
        send_buffer = mpi_cache.mpi_send_buffers[rank]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[rank])
            first = (index - 1) * data_size + 1
            last = (index - 1) * data_size + data_size

            if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[2, :, :,
                                                                             interface])
            else # local element in negative direction
                @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[1, :, :,
                                                                             interface])
            end
        end

        # Each mortar has a total size of 4 * data_size, set everything to NaN first and overwrite the
        # parts where local data exists
        interfaces_data_size = length(mpi_cache.mpi_neighbor_interfaces[rank]) *
                               data_size
        mortars_data_size = length(mpi_cache.mpi_neighbor_mortars[rank]) * 4 * data_size
        send_buffer[(interfaces_data_size + 1):(interfaces_data_size + mortars_data_size)] .= NaN

        for (index, mortar) in enumerate(mpi_cache.mpi_neighbor_mortars[rank])
            # First and last indices in the send buffer for mortar data obtained from local element
            # in a given position
            index_base = interfaces_data_size + (index - 1) * 4 * data_size
            indices = (
                       # first, last for local element in position 1 (lower element)
                       (index_base + 1,
                        index_base + 1 * data_size),
                       # first, last for local element in position 2 (upper element)
                       (index_base + 1 * data_size + 1,
                        index_base + 2 * data_size),
                       # firsts, lasts for local element in position 3 (large element)
                       (index_base + 2 * data_size + 1,
                        index_base + 3 * data_size,
                        index_base + 3 * data_size + 1,
                        index_base + 4 * data_size))

            for position in cache.mpi_mortars.local_neighbor_positions[mortar]
                # Determine whether the data belongs to the left or right side
                if cache.mpi_mortars.large_sides[mortar] == 1 # large element on left side
                    if position in (1, 2) # small element
                        leftright = 2
                    else # large element
                        leftright = 1
                    end
                else # large element on right side
                    if position in (1, 2) # small element
                        leftright = 1
                    else # large element
                        leftright = 2
                    end
                end
                # copy data to buffer
                if position == 1 # lower element
                    first, last = indices[position]
                    @views send_buffer[first:last] .= vec(cache.mpi_mortars.u_lower[leftright,
                                                                                    :,
                                                                                    :,
                                                                                    mortar])
                elseif position == 2 # upper element
                    first, last = indices[position]
                    @views send_buffer[first:last] .= vec(cache.mpi_mortars.u_upper[leftright,
                                                                                    :,
                                                                                    :,
                                                                                    mortar])
                else # large element
                    first_lower, last_lower, first_upper, last_upper = indices[position]
                    @views send_buffer[first_lower:last_lower] .= vec(cache.mpi_mortars.u_lower[leftright,
                                                                                                :,
                                                                                                :,
                                                                                                mortar])
                    @views send_buffer[first_upper:last_upper] .= vec(cache.mpi_mortars.u_upper[leftright,
                                                                                                :,
                                                                                                :,
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

# TODO: MPI dimension agnostic
function finish_mpi_send!(mpi_cache::MPICache)
    MPI.Waitall(mpi_cache.mpi_send_requests, MPI.Status)
end

# TODO: MPI dimension agnostic
function finish_mpi_receive!(mpi_cache::MPICache, mesh, equations, dg, cache)
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    # Start receiving and unpack received data until all communication is finished
    data = MPI.Waitany(mpi_cache.mpi_recv_requests)
    while data !== nothing
        recv_buffer = mpi_cache.mpi_recv_buffers[data]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[data])
            first = (index - 1) * data_size + 1
            last = (index - 1) * data_size + data_size

            if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                @views vec(cache.mpi_interfaces.u[1, :, :, interface]) .= recv_buffer[first:last]
            else # local element in negative direction
                @views vec(cache.mpi_interfaces.u[2, :, :, interface]) .= recv_buffer[first:last]
            end
        end

        interfaces_data_size = length(mpi_cache.mpi_neighbor_interfaces[data]) *
                               data_size
        for (index, mortar) in enumerate(mpi_cache.mpi_neighbor_mortars[data])
            # First and last indices in the receive buffer for mortar data obtained from remote element
            # in a given position
            index_base = interfaces_data_size + (index - 1) * 4 * data_size
            indices = (
                       # first, last for local element in position 1 (lower element)
                       (index_base + 1,
                        index_base + 1 * data_size),
                       # first, last for local element in position 2 (upper element)
                       (index_base + 1 * data_size + 1,
                        index_base + 2 * data_size),
                       # firsts, lasts for local element in position 3 (large element)
                       (index_base + 2 * data_size + 1,
                        index_base + 3 * data_size,
                        index_base + 3 * data_size + 1,
                        index_base + 4 * data_size))

            for position in 1:3
                # Skip if received data for `pos` is NaN as no real data has been sent for the
                # corresponding element
                if isnan(recv_buffer[Base.first(indices[position])])
                    continue
                end

                # Determine whether the received data belongs to the left or right side
                if cache.mpi_mortars.large_sides[mortar] == 1 # large element on left side
                    if position in (1, 2) # small element
                        leftright = 2
                    else # large element
                        leftright = 1
                    end
                else # large element on right side
                    if position in (1, 2) # small element
                        leftright = 1
                    else # large element
                        leftright = 2
                    end
                end

                if position == 1 # lower element data has been received
                    first, last = indices[position]
                    @views vec(cache.mpi_mortars.u_lower[leftright, :, :, mortar]) .= recv_buffer[first:last]
                elseif position == 2 # upper element data has been received
                    first, last = indices[position]
                    @views vec(cache.mpi_mortars.u_upper[leftright, :, :, mortar]) .= recv_buffer[first:last]
                else # large element data has been received
                    first_lower, last_lower, first_upper, last_upper = indices[position]
                    @views vec(cache.mpi_mortars.u_lower[leftright, :, :, mortar]) .= recv_buffer[first_lower:last_lower]
                    @views vec(cache.mpi_mortars.u_upper[leftright, :, :, mortar]) .= recv_buffer[first_upper:last_upper]
                end
            end
        end

        data = MPI.Waitany(mpi_cache.mpi_recv_requests)
    end

    return nothing
end

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::ParallelTreeMesh{2}, equations,
                      dg::DG, RealT, ::Type{uEltype}) where {uEltype <: Real}
    # Get cells for which an element needs to be created (i.e. all leaf cells)
    leaf_cell_ids = local_leaf_cells(mesh.tree)

    elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

    interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

    mpi_interfaces = init_mpi_interfaces(leaf_cell_ids, mesh, elements)

    boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

    mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    mpi_mortars = init_mpi_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

    mpi_cache = init_mpi_cache(mesh, elements, mpi_interfaces, mpi_mortars,
                               nvariables(equations), nnodes(dg), uEltype)

    cache = (; elements, interfaces, mpi_interfaces, boundaries, mortars, mpi_mortars,
             mpi_cache)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
    cache = (; cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

    return cache
end

function init_mpi_cache(mesh, elements, mpi_interfaces, mpi_mortars, nvars, nnodes,
                        uEltype)
    mpi_cache = MPICache(uEltype)

    init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars, nvars,
                    nnodes, uEltype)
    return mpi_cache
end

function init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars, nvars,
                         nnodes, uEltype)
    mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars = init_mpi_neighbor_connectivity(elements,
                                                                                                       mpi_interfaces,
                                                                                                       mpi_mortars,
                                                                                                       mesh)

    mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests = init_mpi_data_structures(mpi_neighbor_interfaces,
                                                                                                        mpi_neighbor_mortars,
                                                                                                        ndims(mesh),
                                                                                                        nvars,
                                                                                                        nnodes,
                                                                                                        uEltype)

    # Determine local and total number of elements
    n_elements_by_rank = Vector{Int}(undef, mpi_nranks())
    n_elements_by_rank[mpi_rank() + 1] = nelements(elements)
    MPI.Allgather!(MPI.UBuffer(n_elements_by_rank, 1), mpi_comm())
    n_elements_by_rank = OffsetArray(n_elements_by_rank, 0:(mpi_nranks() - 1))
    n_elements_global = MPI.Allreduce(nelements(elements), +, mpi_comm())
    @assert n_elements_global==sum(n_elements_by_rank) "error in total number of elements"

    # Determine the global element id of the first element
    first_element_global_id = MPI.Exscan(nelements(elements), +, mpi_comm())
    if mpi_isroot()
        # With Exscan, the result on the first rank is undefined
        first_element_global_id = 1
    else
        # On all other ranks we need to add one, since Julia has one-based indices
        first_element_global_id += 1
    end
    # TODO reuse existing structures
    @pack! mpi_cache = mpi_neighbor_ranks, mpi_neighbor_interfaces,
                       mpi_neighbor_mortars,
                       mpi_send_buffers, mpi_recv_buffers,
                       mpi_send_requests, mpi_recv_requests,
                       n_elements_by_rank, n_elements_global,
                       first_element_global_id
end

# Initialize connectivity between MPI neighbor ranks
function init_mpi_neighbor_connectivity(elements, mpi_interfaces, mpi_mortars,
                                        mesh::TreeMesh2D)
    tree = mesh.tree

    # Determine neighbor ranks and sides for MPI interfaces
    neighbor_ranks_interface = fill(-1, nmpiinterfaces(mpi_interfaces))
    # The global interface id is the smaller of the (globally unique) neighbor cell ids, multiplied by
    # number of directions (2 * ndims) plus direction minus one
    global_interface_ids = fill(-1, nmpiinterfaces(mpi_interfaces))
    for interface_id in 1:nmpiinterfaces(mpi_interfaces)
        orientation = mpi_interfaces.orientations[interface_id]
        remote_side = mpi_interfaces.remote_sides[interface_id]
        # Direction is from local cell to remote cell
        if orientation == 1 # MPI interface in x-direction
            if remote_side == 1 # remote cell on the "left" of MPI interface
                direction = 1
            else # remote cell on the "right" of MPI interface
                direction = 2
            end
        else # MPI interface in y-direction
            if remote_side == 1 # remote cell on the "left" of MPI interface
                direction = 3
            else # remote cell on the "right" of MPI interface
                direction = 4
            end
        end
        local_neighbor_id = mpi_interfaces.local_neighbor_ids[interface_id]
        local_cell_id = elements.cell_ids[local_neighbor_id]
        remote_cell_id = tree.neighbor_ids[direction, local_cell_id]
        neighbor_ranks_interface[interface_id] = tree.mpi_ranks[remote_cell_id]
        if local_cell_id < remote_cell_id
            global_interface_ids[interface_id] = 2 * ndims(tree) * local_cell_id +
                                                 direction - 1
        else
            global_interface_ids[interface_id] = (2 * ndims(tree) * remote_cell_id +
                                                  opposite_direction(direction) - 1)
        end
    end

    # Determine neighbor ranks for MPI mortars
    neighbor_ranks_mortar = Vector{Vector{Int}}(undef, nmpimortars(mpi_mortars))
    # The global mortar id is the (globally unique) large cell id, multiplied by
    # number of directions (2 * ndims) plus direction minus one where
    # direction = 1 for mortars in x-direction where large element is left
    # direction = 2 for mortars in x-direction where large element is right
    # direction = 3 for mortars in y-direction where large element is left
    # direction = 4 for mortars in y-direction where large element is right
    global_mortar_ids = fill(-1, nmpimortars(mpi_mortars))
    for mortar in 1:nmpimortars(mpi_mortars)
        neighbor_ranks_mortar[mortar] = Vector{Int}()

        orientation = mpi_mortars.orientations[mortar]
        large_side = mpi_mortars.large_sides[mortar]
        direction = (orientation - 1) * 2 + large_side

        local_neighbor_ids = mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = mpi_mortars.local_neighbor_positions[mortar]
        if 3 in local_neighbor_positions # large element is on this rank
            large_element_id = local_neighbor_ids[findfirst(pos -> pos == 3,
                                                            local_neighbor_positions)]
            large_cell_id = elements.cell_ids[large_element_id]
        else # large element is remote
            cell_id = elements.cell_ids[first(local_neighbor_ids)]
            large_cell_id = tree.neighbor_ids[direction, tree.parent_ids[cell_id]]
        end

        neighbor_cell_id = tree.neighbor_ids[opposite_direction(direction),
                                             large_cell_id]
        if direction == 1
            lower_cell_id = tree.child_ids[1, neighbor_cell_id]
            upper_cell_id = tree.child_ids[3, neighbor_cell_id]
        elseif direction == 2
            lower_cell_id = tree.child_ids[2, neighbor_cell_id]
            upper_cell_id = tree.child_ids[4, neighbor_cell_id]
        elseif direction == 3
            lower_cell_id = tree.child_ids[1, neighbor_cell_id]
            upper_cell_id = tree.child_ids[2, neighbor_cell_id]
        else
            lower_cell_id = tree.child_ids[3, neighbor_cell_id]
            upper_cell_id = tree.child_ids[4, neighbor_cell_id]
        end

        for cell_id in (lower_cell_id, upper_cell_id, large_cell_id)
            if !is_own_cell(tree, cell_id)
                neighbor_rank = tree.mpi_ranks[cell_id]
                if !(neighbor_rank in neighbor_ranks_mortar[mortar])
                    push!(neighbor_ranks_mortar[mortar], neighbor_rank)
                end
            end
        end

        global_mortar_ids[mortar] = 2 * ndims(tree) * large_cell_id + direction - 1
    end

    # Get sorted, unique neighbor ranks
    mpi_neighbor_ranks = vcat(neighbor_ranks_interface, neighbor_ranks_mortar...) |>
                         sort |> unique

    # Sort interfaces by global interface id
    p = sortperm(global_interface_ids)
    neighbor_ranks_interface .= neighbor_ranks_interface[p]
    interface_ids = collect(1:nmpiinterfaces(mpi_interfaces))[p]

    # Sort mortars by global mortar id
    p = sortperm(global_mortar_ids)
    neighbor_ranks_mortar .= neighbor_ranks_mortar[p]
    mortar_ids = collect(1:nmpimortars(mpi_mortars))[p]

    # For each neighbor rank, init connectivity data structures
    mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
    mpi_neighbor_mortars = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
    for (index, rank) in enumerate(mpi_neighbor_ranks)
        mpi_neighbor_interfaces[index] = interface_ids[findall(x -> (x == rank),
                                                               neighbor_ranks_interface)]
        mpi_neighbor_mortars[index] = mortar_ids[findall(x -> (rank in x),
                                                         neighbor_ranks_mortar)]
    end

    # Sanity checks that we counted all interfaces exactly once
    @assert sum(length(v) for v in mpi_neighbor_interfaces) ==
            nmpiinterfaces(mpi_interfaces)

    return mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars
end

function rhs!(du, u, t,
              mesh::Union{ParallelTreeMesh{2}, ParallelP4estMesh{2},
                          ParallelT8codeMesh{2}}, equations,
              boundary_conditions, source_terms::Source,
              dg::DG, cache) where {Source}
    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(cache, u, mesh, equations, dg.surface_integral, dg)
    end

    # Prolong solution to MPI mortars
    @trixi_timeit timer() "prolong2mpimortars" begin
        prolong2mpimortars!(cache, u, mesh, equations,
                            dg.mortar, dg.surface_integral, dg)
    end

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Prolong solution to interfaces
    # TODO: Taal decide order of arguments, consistent vs. modified cache first?
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Prolong solution to mortars
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u, mesh, equations,
                         dg.mortar, dg.surface_integral, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                          have_nonconservative_terms(equations), equations,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit timer() "MPI interface flux" begin
        calc_mpi_interface_flux!(cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
    end

    # Calculate MPI mortar fluxes
    @trixi_timeit timer() "MPI mortar flux" begin
        calc_mpi_mortar_flux!(cache.elements.surface_flux_values, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.mortar, dg.surface_integral, dg, cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache)
    end

    # Finish to send MPI data
    @trixi_timeit timer() "finish MPI send" finish_mpi_send!(cache.mpi_cache)

    return nothing
end

function prolong2mpiinterfaces!(cache, u,
                                mesh::ParallelTreeMesh{2},
                                equations, surface_integral, dg::DG)
    @unpack mpi_interfaces = cache

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = mpi_interfaces.local_neighbor_ids[interface]

        if mpi_interfaces.orientations[interface] == 1 # interface in x-direction
            if mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                for j in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[2, v, j, interface] = u[v, 1, j, local_element]
                end
            else # local element in negative direction
                for j in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j,
                                                             local_element]
                end
            end
        else # interface in y-direction
            if mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                for i in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[2, v, i, interface] = u[v, i, 1, local_element]
                end
            else # local element in negative direction
                for i in eachnode(dg), v in eachvariable(equations)
                    mpi_interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg),
                                                             local_element]
                end
            end
        end
    end

    return nothing
end

function prolong2mpimortars!(cache, u,
                             mesh::ParallelTreeMesh{2}, equations,
                             mortar_l2::LobattoLegendreMortarL2, surface_integral,
                             dg::DGSEM)
    @unpack mpi_mortars = cache

    @threaded for mortar in eachmpimortar(dg, cache)
        local_neighbor_ids = mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = mpi_mortars.local_neighbor_positions[mortar]

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position in (1, 2) # Current element is small
                # Copy solution small to small
                if mpi_mortars.large_sides[mortar] == 1 # -> small elements on right side
                    if mpi_mortars.orientations[mortar] == 1
                        # L2 mortars in x-direction
                        if position == 1
                            for l in eachnode(dg)
                                for v in eachvariable(equations)
                                    mpi_mortars.u_lower[2, v, l, mortar] = u[v, 1, l,
                                                                             element]
                                end
                            end
                        else # position == 2
                            for l in eachnode(dg)
                                for v in eachvariable(equations)
                                    mpi_mortars.u_upper[2, v, l, mortar] = u[v, 1, l,
                                                                             element]
                                end
                            end
                        end
                    else
                        # L2 mortars in y-direction
                        if position == 1
                            for l in eachnode(dg)
                                for v in eachvariable(equations)
                                    mpi_mortars.u_lower[2, v, l, mortar] = u[v, l, 1,
                                                                             element]
                                end
                            end
                        else # position == 2
                            for l in eachnode(dg)
                                for v in eachvariable(equations)
                                    mpi_mortars.u_upper[2, v, l, mortar] = u[v, l, 1,
                                                                             element]
                                end
                            end
                        end
                    end
                else # large_sides[mortar] == 2 -> small elements on left side
                    if mpi_mortars.orientations[mortar] == 1
                        # L2 mortars in x-direction
                        if position == 1
                            for l in eachnode(dg)
                                for v in eachvariable(equations)
                                    mpi_mortars.u_lower[1, v, l, mortar] = u[v,
                                                                             nnodes(dg),
                                                                             l, element]
                                end
                            end
                        else # position == 2
                            for l in eachnode(dg)
                                for v in eachvariable(equations)
                                    mpi_mortars.u_upper[1, v, l, mortar] = u[v,
                                                                             nnodes(dg),
                                                                             l, element]
                                end
                            end
                        end
                    else
                        # L2 mortars in y-direction
                        if position == 1
                            for l in eachnode(dg)
                                for v in eachvariable(equations)
                                    mpi_mortars.u_lower[1, v, l, mortar] = u[v, l,
                                                                             nnodes(dg),
                                                                             element]
                                end
                            end
                        else # position == 2
                            for l in eachnode(dg)
                                for v in eachvariable(equations)
                                    mpi_mortars.u_upper[1, v, l, mortar] = u[v, l,
                                                                             nnodes(dg),
                                                                             element]
                                end
                            end
                        end
                    end
                end
            else # position == 3 -> current element is large
                # Interpolate large element face data to small interface locations
                if mpi_mortars.large_sides[mortar] == 1 # -> large element on left side
                    leftright = 1
                    if mpi_mortars.orientations[mortar] == 1
                        # L2 mortars in x-direction
                        u_large = view(u, :, nnodes(dg), :, element)
                        element_solutions_to_mortars!(mpi_mortars, mortar_l2, leftright,
                                                      mortar, u_large)
                    else
                        # L2 mortars in y-direction
                        u_large = view(u, :, :, nnodes(dg), element)
                        element_solutions_to_mortars!(mpi_mortars, mortar_l2, leftright,
                                                      mortar, u_large)
                    end
                else # large_sides[mortar] == 2 -> large element on right side
                    leftright = 2
                    if mpi_mortars.orientations[mortar] == 1
                        # L2 mortars in x-direction
                        u_large = view(u, :, 1, :, element)
                        element_solutions_to_mortars!(mpi_mortars, mortar_l2, leftright,
                                                      mortar, u_large)
                    else
                        # L2 mortars in y-direction
                        u_large = view(u, :, :, 1, element)
                        element_solutions_to_mortars!(mpi_mortars, mortar_l2, leftright,
                                                      mortar, u_large)
                    end
                end
            end
        end
    end

    return nothing
end

function calc_mpi_interface_flux!(surface_flux_values,
                                  mesh::ParallelTreeMesh{2},
                                  nonconservative_terms::False, equations,
                                  surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u, local_neighbor_ids, orientations, remote_sides = cache.mpi_interfaces

    @threaded for interface in eachmpiinterface(dg, cache)
        # Get local neighboring element
        element = local_neighbor_ids[interface]

        # Determine interface direction with respect to element:
        if orientations[interface] == 1 # interface in x-direction
            if remote_sides[interface] == 1 # local element in positive direction
                direction = 1
            else # local element in negative direction
                direction = 2
            end
        else # interface in y-direction
            if remote_sides[interface] == 1 # local element in positive direction
                direction = 3
            else # local element in negative direction
                direction = 4
            end
        end

        for i in eachnode(dg)
            # Call pointwise Riemann solver
            u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
            flux = surface_flux(u_ll, u_rr, orientations[interface], equations)

            # Copy flux to local element storage
            for v in eachvariable(equations)
                surface_flux_values[v, i, direction, element] = flux[v]
            end
        end
    end

    return nothing
end

function calc_mpi_mortar_flux!(surface_flux_values,
                               mesh::ParallelTreeMesh{2},
                               nonconservative_terms::False, equations,
                               mortar_l2::LobattoLegendreMortarL2,
                               surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack u_lower, u_upper, orientations = cache.mpi_mortars
    @unpack fstar_upper_threaded, fstar_lower_threaded = cache

    @threaded for mortar in eachmpimortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar_upper = fstar_upper_threaded[Threads.threadid()]
        fstar_lower = fstar_lower_threaded[Threads.threadid()]

        # Calculate fluxes
        orientation = orientations[mortar]
        calc_fstar!(fstar_upper, equations, surface_flux, dg, u_upper, mortar,
                    orientation)
        calc_fstar!(fstar_lower, equations, surface_flux, dg, u_lower, mortar,
                    orientation)

        mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                       mesh, equations, mortar_l2, dg, cache,
                                       mortar, fstar_upper, fstar_lower)
    end

    return nothing
end

@inline function mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                                mesh::ParallelTreeMesh{2}, equations,
                                                mortar_l2::LobattoLegendreMortarL2,
                                                dg::DGSEM, cache,
                                                mortar, fstar_upper, fstar_lower)
    local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
    local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

    for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
        if position in (1, 2) # Current element is small
            # Copy flux small to small
            if cache.mpi_mortars.large_sides[mortar] == 1 # -> small elements on right side
                if cache.mpi_mortars.orientations[mortar] == 1
                    # L2 mortars in x-direction
                    direction = 1
                else
                    # L2 mortars in y-direction
                    direction = 3
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if cache.mpi_mortars.orientations[mortar] == 1
                    # L2 mortars in x-direction
                    direction = 2
                else
                    # L2 mortars in y-direction
                    direction = 4
                end
            end

            if position == 1
                surface_flux_values[:, :, direction, element] .= fstar_lower
            elseif position == 2
                surface_flux_values[:, :, direction, element] .= fstar_upper
            end
        else # position == 3 -> current element is large
            # Project small fluxes to large element
            if cache.mpi_mortars.large_sides[mortar] == 1 # -> large element on left side
                if cache.mpi_mortars.orientations[mortar] == 1
                    # L2 mortars in x-direction
                    direction = 2
                else
                    # L2 mortars in y-direction
                    direction = 4
                end
            else # large_sides[mortar] == 2 -> large element on right side
                if cache.mpi_mortars.orientations[mortar] == 1
                    # L2 mortars in x-direction
                    direction = 1
                else
                    # L2 mortars in y-direction
                    direction = 3
                end
            end

            multiply_dimensionwise!(view(surface_flux_values, :, :, direction, element),
                                    mortar_l2.reverse_upper, fstar_upper,
                                    mortar_l2.reverse_lower, fstar_lower)
        end
    end

    return nothing
end
end # @muladd
