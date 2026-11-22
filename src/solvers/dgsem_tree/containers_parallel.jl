# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Data structures for the MPI communication, valid for all dimensions
mutable struct TreeMPICache{uEltype <: Real}
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

function TreeMPICache(uEltype)
    # MPI communication "just works" for bitstypes only
    if !isbitstype(uEltype)
        throw(ArgumentError("TreeMPICache only supports bitstypes, $uEltype is not a bitstype."))
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

    return TreeMPICache{uEltype}(mpi_neighbor_ranks,
                                 mpi_neighbor_interfaces,
                                 mpi_neighbor_mortars,
                                 mpi_send_buffers, mpi_recv_buffers,
                                 mpi_send_requests, mpi_recv_requests,
                                 n_elements_by_rank, n_elements_global,
                                 first_element_global_id)
end
@inline Base.eltype(::TreeMPICache{uEltype}) where {uEltype} = uEltype

function init_mpi_cache(mesh, elements, mpi_interfaces, mpi_mortars,
                        nvars, nnodes, uEltype)
    mpi_cache = TreeMPICache(uEltype)

    init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars,
                    nvars, nnodes, uEltype)
    return mpi_cache
end

function init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, mpi_mortars,
                         nvars, nnodes, uEltype)
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

    @pack! mpi_cache = mpi_neighbor_ranks, mpi_neighbor_interfaces,
                       mpi_neighbor_mortars,
                       mpi_send_buffers, mpi_recv_buffers,
                       mpi_send_requests, mpi_recv_requests,
                       n_elements_by_rank, n_elements_global,
                       first_element_global_id
end
end # @muladd
