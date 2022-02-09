# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


mutable struct P4estMPICache{uEltype}
  mpi_neighbor_ranks::Vector{Int}
  mpi_neighbor_interfaces::Vector{Vector{Int}}
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
  mpi_send_buffers = Vector{Vector{uEltype}}(undef, 0)
  mpi_recv_buffers = Vector{Vector{uEltype}}(undef, 0)
  mpi_send_requests = Vector{MPI.Request}(undef, 0)
  mpi_recv_requests = Vector{MPI.Request}(undef, 0)
  n_elements_by_rank = OffsetArray(Vector{Int}(undef, 0), 0:-1)
  n_elements_global = 0
  first_element_global_id = 0

  P4estMPICache{uEltype}(mpi_neighbor_ranks, mpi_neighbor_interfaces,
                         mpi_send_buffers, mpi_recv_buffers,
                         mpi_send_requests, mpi_recv_requests,
                         n_elements_by_rank, n_elements_global,
                         first_element_global_id)
end


function start_mpi_send!(mpi_cache::P4estMPICache, mesh, equations, dg, cache)
  data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

  for d in 1:length(mpi_cache.mpi_neighbor_ranks)
    send_buffer = mpi_cache.mpi_send_buffers[d]

    for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
      first = (index - 1) * data_size + 1
      last  = (index - 1) * data_size + data_size
      local_side = cache.mpi_interfaces.local_sides[interface]
      @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[local_side, .., interface])
    end
  end

  for (index, d) in enumerate(mpi_cache.mpi_neighbor_ranks)
    mpi_cache.mpi_send_requests[index] = MPI.Isend(
      mpi_cache.mpi_send_buffers[index], d, mpi_rank(), mpi_comm())
  end

  return nothing
end


function start_mpi_receive!(mpi_cache::P4estMPICache)
  for (index, d) in enumerate(mpi_cache.mpi_neighbor_ranks)
    mpi_cache.mpi_recv_requests[index] = MPI.Irecv!(
      mpi_cache.mpi_recv_buffers[index], d, d, mpi_comm())
  end

  return nothing
end


function finish_mpi_send!(mpi_cache::P4estMPICache)
  MPI.Waitall!(mpi_cache.mpi_send_requests)
end


function finish_mpi_receive!(mpi_cache::P4estMPICache, mesh, equations, dg, cache)
  data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

  # Start receiving and unpack received data until all communication is finished
  d, _ = MPI.Waitany!(mpi_cache.mpi_recv_requests)
  while d != 0
    recv_buffer = mpi_cache.mpi_recv_buffers[d]

    for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
      first = (index - 1) * data_size + 1
      last  = (index - 1) * data_size + data_size

      if cache.mpi_interfaces.local_sides[interface] == 1 # local element on primary side
        @views vec(cache.mpi_interfaces.u[2, .., interface]) .= recv_buffer[first:last]
      else # local element at secondary side
        @views vec(cache.mpi_interfaces.u[1, .., interface]) .= recv_buffer[first:last]
      end
    end

    d, _ = MPI.Waitany!(mpi_cache.mpi_recv_requests)
  end

  return nothing
end


# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::ParallelP4estMesh, equations::AbstractEquations, dg::DG, ::Any, ::Type{uEltype}) where {uEltype<:Real}
  # Make sure to balance the p4est before creating any containers
  # in case someone has tampered with the p4est after creating the mesh
  balance!(mesh)

  elements       = init_elements(mesh, equations, dg.basis, uEltype)
  interfaces     = init_interfaces(mesh, equations, dg.basis, elements)
  mpi_interfaces = init_mpi_interfaces(mesh, equations, dg.basis, elements)
  boundaries     = init_boundaries(mesh, equations, dg.basis, elements)
  mortars        = init_mortars(mesh, equations, dg.basis, elements)
  mpi_cache      = init_mpi_cache(mesh, elements, mpi_interfaces,
                                  nvariables(equations), nnodes(dg), uEltype)

  cache = (; elements, interfaces, mpi_interfaces, boundaries, mortars, mpi_cache)

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (; cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
  cache = (; cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

  return cache
end


function init_mpi_cache(mesh::ParallelP4estMesh, elements, mpi_interfaces, nvars, nnodes, uEltype)
  mpi_cache = P4estMPICache(uEltype)
  init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, nvars, nnodes, uEltype)

  return mpi_cache
end

function init_mpi_cache!(mpi_cache::P4estMPICache, mesh::ParallelP4estMesh, elements, mpi_interfaces,
                         nvars, n_nodes, uEltype)
  mpi_neighbor_ranks, mpi_neighbor_interfaces =
    init_mpi_neighbor_connectivity(mpi_interfaces, mesh)

  mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests =
    init_mpi_data_structures(mpi_neighbor_interfaces, ndims(mesh), nvars, n_nodes, uEltype, nothing)

  # Determine local and total number of elements
  n_elements_global = Int(mesh.p4est.global_num_quadrants)
  n_elements_by_rank = vcat(Int.(unsafe_wrap(Array, mesh.p4est.global_first_quadrant, mpi_nranks())),
                            n_elements_global) |> diff # diff sufficient due to 0-based quad indices
  n_elements_by_rank = OffsetArray(n_elements_by_rank, 0:(mpi_nranks() - 1))
  # Account for 1-based indexing in Julia
  first_element_global_id = Int(unsafe_load(mesh.p4est.global_first_quadrant, mpi_rank() + 1)) + 1
  @assert n_elements_global == sum(n_elements_by_rank) "error in total number of elements"

  # TODO reuse existing structures
  @pack! mpi_cache = mpi_neighbor_ranks, mpi_neighbor_interfaces,
                     mpi_send_buffers, mpi_recv_buffers,
                     mpi_send_requests, mpi_recv_requests,
                     n_elements_by_rank, n_elements_global,
                     first_element_global_id

end

function init_mpi_neighbor_connectivity(mpi_interfaces, mesh::ParallelP4estMesh)
  # Let p4est iterate over all interfaces and call init_neighbor_rank_connectivity_iter_face
  # to collect connectivity information
  iter_face_c = cfunction(init_neighbor_rank_connectivity_iter_face, Val(ndims(mesh)))
  user_data = InitNeighborRankConnectivityIterFaceUserData(mpi_interfaces, mesh)

  # Ghost layer is required to determine owner ranks of quadrants on neighboring processes
  ghost_layer = new_ghost_p4est(mesh.p4est)
  @assert ghost_is_valid_p4est(mesh.p4est, ghost_layer) == 1
  iterate_p4est(mesh.p4est, user_data; ghost_layer=ghost_layer, iter_face_c=iter_face_c)
  ghost_destroy_p4est(ghost_layer)

  # Build proper connectivity data structures from information gathered by iterating over p4est
  @unpack global_interface_ids, neighbor_ranks_interface = user_data
  mpi_neighbor_ranks = neighbor_ranks_interface |> sort |> unique
  p = sortperm(global_interface_ids)
  neighbor_ranks_interface .= neighbor_ranks_interface[p]
  interface_ids = collect(1:nmpiinterfaces(mpi_interfaces))[p]

  # For each neighbor rank, init connectivity data structures
  mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
  for (index, d) in enumerate(mpi_neighbor_ranks)
    mpi_neighbor_interfaces[index] = interface_ids[findall(==(d), neighbor_ranks_interface)]
  end

  # Check that all interfaces were counted exactly once
  @assert sum(length(v) for v in mpi_neighbor_interfaces) == nmpiinterfaces(mpi_interfaces)

  return mpi_neighbor_ranks, mpi_neighbor_interfaces
end

mutable struct InitNeighborRankConnectivityIterFaceUserData{MPIInterfaces, Mesh}
  interfaces::MPIInterfaces
  interface_id::Int
  global_interface_ids::Vector{Int}
  neighbor_ranks_interface::Vector{Int}
  mesh::Mesh
end

function InitNeighborRankConnectivityIterFaceUserData(mpi_interfaces, mesh)
  global_interface_ids = fill(-1, nmpiinterfaces(mpi_interfaces))
  neighbor_ranks_interface = fill(-1, nmpiinterfaces(mpi_interfaces))

  return InitNeighborRankConnectivityIterFaceUserData{typeof(mpi_interfaces), typeof(mesh)}(
    mpi_interfaces, 1, global_interface_ids, neighbor_ranks_interface, mesh)
end

function init_neighbor_rank_connectivity_iter_face(info, user_data)
  data = unsafe_pointer_to_objref(Ptr{InitNeighborRankConnectivityIterFaceUserData}(user_data))

  # Function barrier because the unpacked user_data above is not type-stable
  init_neighbor_rank_connectivity_iter_face_inner(info, data)
end

# 2D
cfunction(::typeof(init_neighbor_rank_connectivity_iter_face), ::Val{2}) = @cfunction(init_neighbor_rank_connectivity_iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
# 3D
cfunction(::typeof(init_neighbor_rank_connectivity_iter_face), ::Val{3}) = @cfunction(init_neighbor_rank_connectivity_iter_face, Cvoid, (Ptr{p8est_iter_face_info_t}, Ptr{Cvoid}))

# Function barrier for type stability
function init_neighbor_rank_connectivity_iter_face_inner(info, user_data)
  @unpack interfaces, interface_id, global_interface_ids, neighbor_ranks_interface, mesh = user_data

  # Get the global interface ids and neighbor rank if current face belongs to an MPI interface
  if info.sides.elem_count == 2 # MPI interfaces have two neighboring elements
    # Extract surface data
    sides = (unsafe_load_side(info, 1), unsafe_load_side(info, 2))

    if sides[1].is_hanging == 0 && sides[2].is_hanging == 0 # No hanging nodes for MPI interfaces
      if sides[1].is.full.is_ghost == 1
        remote_side = 1
        local_side = 2
      elseif sides[2].is.full.is_ghost == 1
        remote_side = 2
        local_side = 1
      else # both sides are on this rank -> skip since it's a regular interface
        return nothing
      end

      # Sanity check, current face should belong to current MPI interface
      local_tree = unsafe_load_tree(mesh.p4est, sides[local_side].treeid + 1) # one-based indexing
      local_quad_id = local_tree.quadrants_offset + sides[local_side].is.full.quadid
      @assert interfaces.local_element_ids[interface_id] == local_quad_id + 1 # one-based indexing

      # Get neighbor ID from ghost layer
      proc_offsets = unsafe_wrap(Array, info.ghost_layer.proc_offsets, mpi_nranks() + 1)
      ghost_id = sides[remote_side].is.full.quadid # indexes the ghost layer, 0-based
      neighbor_rank = findfirst(r -> proc_offsets[r] <= ghost_id < proc_offsets[r+1],
                                1:mpi_nranks()) - 1 # mpi ranks are 0-based
      neighbor_ranks_interface[interface_id] = neighbor_rank

      # Global interface id is the globally unique quadrant id of the quadrant on the primary
      # side (1) multiplied by the number of faces per quadrant plus face
      if local_side == 1
        offset = unsafe_load(mesh.p4est.global_first_quadrant, mpi_rank() + 1) # one-based indexing
        primary_quad_id = offset + local_quad_id
      else
        offset = unsafe_load(mesh.p4est.global_first_quadrant, neighbor_rank + 1) # one-based indexing
        primary_quad_id = offset + sides[1].is.full.quad.p.piggy3.local_num
      end
      global_interface_id = 2 * ndims(mesh) * primary_quad_id + sides[1].face
      global_interface_ids[interface_id] = global_interface_id

      user_data.interface_id += 1
    end
  end

  return nothing
end

# TODO: ::Any is a temporary hack. This method is already defined in the TreeMesh
# code. It uses `mpi_neighbor_mortars` as an argument but not uEltype, thus the number
# of arguments would be the same without ::Any which would break precompilation.
# See https://github.com/trixi-framework/Trixi.jl/pull/977#discussion_r793694635 for further
# discussion.
function init_mpi_data_structures(mpi_neighbor_interfaces, n_dims, nvars, n_nodes, uEltype, ::Any)
  data_size = nvars * n_nodes^(n_dims - 1)
  mpi_send_buffers = Vector{Vector{uEltype}}(undef, length(mpi_neighbor_interfaces))
  mpi_recv_buffers = Vector{Vector{uEltype}}(undef, length(mpi_neighbor_interfaces))
  for index in 1:length(mpi_neighbor_interfaces)
    mpi_send_buffers[index] = Vector{uEltype}(undef, length(mpi_neighbor_interfaces[index]) * data_size)
    mpi_recv_buffers[index] = Vector{uEltype}(undef, length(mpi_neighbor_interfaces[index]) * data_size)
  end

  mpi_send_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_interfaces))
  mpi_recv_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_interfaces))

  return mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests
end


include("dg_2d_parallel.jl")

end # muladd