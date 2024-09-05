@muladd begin
#! format: noindent

# This method is called when a `SemidiscretizationHyperbolic` is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::ParallelT8codeMesh, equations::AbstractEquations, dg::DG,
                      ::Any,
                      ::Type{uEltype}) where {uEltype <: Real}
    # Make sure to balance and partition the forest before creating any
    # containers in case someone has tampered with forest after creating the
    # mesh.
    balance!(mesh)
    partition!(mesh)

    count_required_surfaces!(mesh)

    elements = init_elements(mesh, equations, dg.basis, uEltype)
    mortars = init_mortars(mesh, equations, dg.basis, elements)
    interfaces = init_interfaces(mesh, equations, dg.basis, elements)
    boundaries = init_boundaries(mesh, equations, dg.basis, elements)

    mpi_mortars = init_mpi_mortars(mesh, equations, dg.basis, elements)
    mpi_interfaces = init_mpi_interfaces(mesh, equations, dg.basis, elements)

    mpi_mesh_info = (mpi_mortars = mpi_mortars,
                     mpi_interfaces = mpi_interfaces,
                     global_mortar_ids = fill(UInt64(0), nmpimortars(mpi_mortars)),
                     global_interface_ids = fill(UInt64(0),
                                                 nmpiinterfaces(mpi_interfaces)),
                     neighbor_ranks_mortar = Vector{Vector{Int}}(undef,
                                                                 nmpimortars(mpi_mortars)),
                     neighbor_ranks_interface = fill(-1,
                                                     nmpiinterfaces(mpi_interfaces)))

    fill_mesh_info!(mesh, interfaces, mortars, boundaries,
                    mesh.boundary_names; mpi_mesh_info = mpi_mesh_info)

    mpi_cache = init_mpi_cache(mesh, mpi_mesh_info, nvariables(equations), nnodes(dg),
                               uEltype)

    empty!(mpi_mesh_info.global_mortar_ids)
    empty!(mpi_mesh_info.global_interface_ids)
    empty!(mpi_mesh_info.neighbor_ranks_mortar)
    empty!(mpi_mesh_info.neighbor_ranks_interface)

    init_normal_directions!(mpi_mortars, dg.basis, elements)
    exchange_normal_directions!(mpi_mortars, mpi_cache, mesh, nnodes(dg))

    cache = (; elements, interfaces, mpi_interfaces, boundaries, mortars, mpi_mortars,
             mpi_cache)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
    cache = (; cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

    return cache
end

function init_mpi_cache(mesh::ParallelT8codeMesh, mpi_mesh_info, nvars, nnodes, uEltype)
    mpi_cache = P4estMPICache(uEltype)
    init_mpi_cache!(mpi_cache, mesh, mpi_mesh_info, nvars, nnodes, uEltype)
    return mpi_cache
end

function init_mpi_cache!(mpi_cache::P4estMPICache, mesh::ParallelT8codeMesh,
                         mpi_mesh_info, nvars, nnodes, uEltype)
    mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars = init_mpi_neighbor_connectivity(mpi_mesh_info,
                                                                                                       mesh)

    mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests = init_mpi_data_structures(mpi_neighbor_interfaces,
                                                                                                        mpi_neighbor_mortars,
                                                                                                        ndims(mesh),
                                                                                                        nvars,
                                                                                                        nnodes,
                                                                                                        uEltype)

    n_elements_global = Int(t8_forest_get_global_num_elements(mesh.forest))
    n_elements_local = Int(t8_forest_get_local_num_elements(mesh.forest))

    n_elements_by_rank = Vector{Int}(undef, mpi_nranks())
    n_elements_by_rank[mpi_rank() + 1] = n_elements_local

    MPI.Allgather!(MPI.UBuffer(n_elements_by_rank, 1), mpi_comm())

    n_elements_by_rank = OffsetArray(n_elements_by_rank, 0:(mpi_nranks() - 1))

    # Account for 1-based indexing in Julia.
    first_element_global_id = sum(n_elements_by_rank[0:(mpi_rank() - 1)]) + 1

    @assert n_elements_global==sum(n_elements_by_rank) "error in total number of elements"

    @pack! mpi_cache = mpi_neighbor_ranks, mpi_neighbor_interfaces,
                       mpi_neighbor_mortars,
                       mpi_send_buffers, mpi_recv_buffers,
                       mpi_send_requests, mpi_recv_requests,
                       n_elements_by_rank, n_elements_global,
                       first_element_global_id

    return mpi_cache
end

function init_mpi_neighbor_connectivity(mpi_mesh_info, mesh::ParallelT8codeMesh)
    @unpack mpi_interfaces, mpi_mortars, global_interface_ids, neighbor_ranks_interface, global_mortar_ids, neighbor_ranks_mortar = mpi_mesh_info

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
end # @muladd
