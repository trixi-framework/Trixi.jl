# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# everything related to a DG semidiscretization in 2D using MPI,
# currently limited to Lobatto-Legendre nodes


# TODO: MPI dimension agnostic
# TODO: MPI, adapt to different real types (and AD!)
mutable struct MPICache
  mpi_neighbor_ranks::Vector{Int}
  mpi_neighbor_interfaces::Vector{Vector{Int}}
  mpi_send_buffers::Vector{Vector{Float64}}
  mpi_recv_buffers::Vector{Vector{Float64}}
  mpi_send_requests::Vector{MPI.Request}
  mpi_recv_requests::Vector{MPI.Request}
  n_elements_by_rank::OffsetArray{Int, 1, Array{Int, 1}}
  n_elements_global::Int
  first_element_global_id::Int
end


function MPICache()
  mpi_neighbor_ranks = Vector{Int}(undef, 0)
  mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, 0)
  mpi_send_buffers = Vector{Vector{Float64}}(undef, 0)
  mpi_recv_buffers = Vector{Vector{Float64}}(undef, 0)
  mpi_send_requests = Vector{MPI.Request}(undef, 0)
  mpi_recv_requests = Vector{MPI.Request}(undef, 0)
  n_elements_by_rank = OffsetArray(Vector{Int}(undef, 0), 0:-1)
  n_elements_global = 0
  first_element_global_id = 0

  MPICache(mpi_neighbor_ranks, mpi_neighbor_interfaces,
           mpi_send_buffers, mpi_recv_buffers,
           mpi_send_requests, mpi_recv_requests,
           n_elements_by_rank, n_elements_global,
           first_element_global_id)
end


# TODO: MPI dimension agnostic
function start_mpi_receive!(mpi_cache::MPICache)

  for (index, d) in enumerate(mpi_cache.mpi_neighbor_ranks)
    mpi_cache.mpi_recv_requests[index] = MPI.Irecv!(
      mpi_cache.mpi_recv_buffers[index], d, d, mpi_comm())
  end

  return nothing
end


# TODO: MPI dimension agnostic
function start_mpi_send!(mpi_cache::MPICache, mesh, equations, dg, cache)
  data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

  for d in 1:length(mpi_cache.mpi_neighbor_ranks)
    send_buffer = mpi_cache.mpi_send_buffers[d]

    for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
      first = (index - 1) * data_size + 1
      last =  (index - 1) * data_size + data_size

      if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
        @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[2, :, :, interface])
      else # local element in negative direction
        @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[1, :, :, interface])
      end
    end
  end

  # Start sending
  for (index, d) in enumerate(mpi_cache.mpi_neighbor_ranks)
    mpi_cache.mpi_send_requests[index] = MPI.Isend(
      mpi_cache.mpi_send_buffers[index], d, mpi_rank(), mpi_comm())
  end

  return nothing
end


# TODO: MPI dimension agnostic
function finish_mpi_send!(mpi_cache::MPICache)
  MPI.Waitall!(mpi_cache.mpi_send_requests)
end


# TODO: MPI dimension agnostic
function finish_mpi_receive!(mpi_cache::MPICache, mesh, equations, dg, cache)
  data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

  # Start receiving and unpack received data until all communication is finished
  d, _ = MPI.Waitany!(mpi_cache.mpi_recv_requests)
  while d != 0
    recv_buffer = mpi_cache.mpi_recv_buffers[d]

    for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[d])
      first = (index - 1) * data_size + 1
      last =  (index - 1) * data_size + data_size

      if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
        @views vec(cache.mpi_interfaces.u[1, :, :, interface]) .= recv_buffer[first:last]
      else # local element in negative direction
        @views vec(cache.mpi_interfaces.u[2, :, :, interface]) .= recv_buffer[first:last]
      end
    end

    d, _ = MPI.Waitany!(mpi_cache.mpi_recv_requests)
  end

  return nothing
end


# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::ParallelTreeMesh{2}, equations,
                      dg::DG, RealT, uEltype)
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  elements = init_elements(leaf_cell_ids, mesh, equations, dg.basis, RealT, uEltype)

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

  mpi_interfaces = init_mpi_interfaces(leaf_cell_ids, mesh, elements)

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

  mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

  mpi_cache = init_mpi_cache(mesh, elements, mpi_interfaces, nvariables(equations), nnodes(dg))

  cache = (; elements, interfaces, mpi_interfaces, boundaries, mortars,
             mpi_cache)

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)
  cache = (;cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

  return cache
end


function init_mpi_cache(mesh, elements, mpi_interfaces, nvars, nnodes)
  mpi_cache = MPICache()

  init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, nvars, nnodes)
  return mpi_cache
end


function init_mpi_cache!(mpi_cache, mesh, elements, mpi_interfaces, nvars, nnodes)
  mpi_neighbor_ranks, mpi_neighbor_interfaces =
    init_mpi_neighbor_connectivity(elements, mpi_interfaces, mesh)

  mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests =
    init_mpi_data_structures(mpi_neighbor_interfaces, ndims(mesh), nvars, nnodes)

  # Determine local and total number of elements
  n_elements_by_rank = Vector{Int}(undef, mpi_nranks())
  n_elements_by_rank[mpi_rank() + 1] = nelements(elements)
  MPI.Allgather!(MPI.UBuffer(n_elements_by_rank, 1), mpi_comm())
  n_elements_by_rank = OffsetArray(n_elements_by_rank, 0:(mpi_nranks() - 1))
  n_elements_global = MPI.Allreduce(nelements(elements), +, mpi_comm())
  @assert n_elements_global == sum(n_elements_by_rank) "error in total number of elements"

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
                     mpi_send_buffers, mpi_recv_buffers,
                     mpi_send_requests, mpi_recv_requests,
                     n_elements_by_rank, n_elements_global,
                     first_element_global_id
end


# Initialize connectivity between MPI neighbor ranks
function init_mpi_neighbor_connectivity(elements, mpi_interfaces, mesh::TreeMesh2D)
  tree = mesh.tree

  # Determine neighbor ranks and sides for MPI interfaces
  neighbor_ranks = fill(-1, nmpiinterfaces(mpi_interfaces))
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
    local_element_id = mpi_interfaces.local_element_ids[interface_id]
    local_cell_id = elements.cell_ids[local_element_id]
    remote_cell_id = tree.neighbor_ids[direction, local_cell_id]
    neighbor_ranks[interface_id] = tree.mpi_ranks[remote_cell_id]
    if local_cell_id < remote_cell_id
      global_interface_ids[interface_id] = 2 * ndims(tree) * local_cell_id + direction - 1
    else
      global_interface_ids[interface_id] = (2 * ndims(tree) * remote_cell_id +
                                            opposite_direction(direction) - 1)
    end
  end

  # Get sorted, unique neighbor ranks
  mpi_neighbor_ranks = unique(sort(neighbor_ranks))

  # Sort interfaces by global interface id
  p = sortperm(global_interface_ids)
  neighbor_ranks .= neighbor_ranks[p]
  interface_ids = collect(1:nmpiinterfaces(mpi_interfaces))[p]

  # For each neighbor rank, init connectivity data structures
  mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
  for (index, d) in enumerate(mpi_neighbor_ranks)
    mpi_neighbor_interfaces[index] = interface_ids[findall(x->(x == d), neighbor_ranks)]
  end

  # Sanity check that we counted all interfaces exactly once
  @assert sum(length(v) for v in mpi_neighbor_interfaces) == nmpiinterfaces(mpi_interfaces)

  return mpi_neighbor_ranks, mpi_neighbor_interfaces
end


# TODO: MPI dimension agnostic
# Initialize MPI data structures
function init_mpi_data_structures(mpi_neighbor_interfaces, ndims, nvars, n_nodes)
  data_size = nvars * n_nodes^(ndims - 1)
  mpi_send_buffers = Vector{Vector{Float64}}(undef, length(mpi_neighbor_interfaces))
  mpi_recv_buffers = Vector{Vector{Float64}}(undef, length(mpi_neighbor_interfaces))
  for index in 1:length(mpi_neighbor_interfaces)
    mpi_send_buffers[index] = Vector{Float64}(undef, length(mpi_neighbor_interfaces[index]) * data_size)
    mpi_recv_buffers[index] = Vector{Float64}(undef, length(mpi_neighbor_interfaces[index]) * data_size)
  end

  mpi_send_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_interfaces))
  mpi_recv_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_interfaces))

  return mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests
end


function rhs!(du, u, t,
              mesh::ParallelTreeMesh{2}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Start to receive MPI data
  @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

  # Prolong solution to MPI interfaces
  @trixi_timeit timer() "prolong2mpiinterfaces" prolong2mpiinterfaces!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Start to send MPI data
  @trixi_timeit timer() "start MPI send" start_mpi_send!(
    cache.mpi_cache, mesh, equations, dg, cache)

  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  # TODO: Taal decide order of arguments, consistent vs. modified cache first?
  @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.surface_integral, dg, cache)

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
    cache, t, boundary_conditions, mesh, equations, dg.surface_integral, dg)

  # Prolong solution to mortars
  @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
    cache, u, mesh, equations, dg.mortar, dg.surface_integral, dg)

  # Calculate mortar fluxes
  @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.mortar, dg.surface_integral, dg, cache)

  # Finish to receive MPI data
  @trixi_timeit timer() "finish MPI receive" finish_mpi_receive!(
    cache.mpi_cache, mesh, equations, dg, cache)

  # Calculate MPI interface fluxes
  @trixi_timeit timer() "MPI interface flux" calc_mpi_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.surface_integral, dg, cache)

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations, dg.surface_integral, dg, cache)

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @trixi_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  # Finish to send MPI data
  @trixi_timeit timer() "finish MPI send" finish_mpi_send!(cache.mpi_cache)

  return nothing
end


function prolong2mpiinterfaces!(cache, u,
                                mesh::ParallelTreeMesh{2},
                                equations, surface_integral, dg::DG)
  @unpack mpi_interfaces = cache

  @threaded for interface in eachmpiinterface(mesh, dg, cache)
    local_element = mpi_interfaces.local_element_ids[interface]

    if mpi_interfaces.orientations[interface] == 1 # interface in x-direction
      if mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
        for j in eachnode(dg), v in eachvariable(equations)
          mpi_interfaces.u[2, v, j, interface] = u[v,          1, j, local_element]
        end
      else # local element in negative direction
        for j in eachnode(dg), v in eachvariable(equations)
          mpi_interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j, local_element]
        end
      end
    else # interface in y-direction
      if mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
        for i in eachnode(dg), v in eachvariable(equations)
          mpi_interfaces.u[2, v, i, interface] = u[v, i,          1, local_element]
        end
      else # local element in negative direction
        for i in eachnode(dg), v in eachvariable(equations)
          mpi_interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), local_element]
        end
      end
    end
  end

  return nothing
end


function calc_mpi_interface_flux!(surface_flux_values,
                                  mesh::ParallelTreeMesh{2},
                                  nonconservative_terms::Val{false}, equations,
                                  surface_integral, dg::DG, cache)
  @unpack surface_flux = surface_integral
  @unpack u, local_element_ids, orientations, remote_sides = cache.mpi_interfaces

  @threaded for interface in eachmpiinterface(mesh, dg, cache)
    # Get local neighboring element
    element = local_element_ids[interface]

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


end # @muladd
