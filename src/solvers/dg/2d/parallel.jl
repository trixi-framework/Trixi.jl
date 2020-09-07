function init_mpi()
  if !MPI.Initialized()
    # MPI.THREAD_FUNNELED: Only main thread makes MPI calls
    provided = MPI.Init_thread(MPI.THREAD_FUNNELED)
    @assert provided >= MPI.THREAD_FUNNELED "MPI library with insufficient threading support"
  end
end


# Count the number of MPI interfaces that need to be created
function count_required_mpi_interfaces(mesh::TreeMesh{2}, cell_ids)
  count = 0

  # Iterate over all cells
  for cell_id in cell_ids
    for direction in 1:n_directions(mesh.tree)
      # If no neighbor exists, current cell is small or at boundary and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is on this domain -> create regular interface instead
      if is_parallel() && is_own_cell(mesh.tree, neighbor_cell_id)
        continue
      end

      count += 1
    end
  end

  return count
end


# Create MPI interface container, initialize interface data, and return interface container for further use
function init_mpi_interfaces(cell_ids, mesh::TreeMesh{2}, ::Val{NVARS}, ::Val{POLYDEG}, elements) where {NVARS, POLYDEG}
  # Initialize container
  n_mpi_interfaces = count_required_mpi_interfaces(mesh, cell_ids)
  mpi_interfaces = MpiInterfaceContainer2D{NVARS, POLYDEG}(n_mpi_interfaces)

  # Connect elements with interfaces
  init_mpi_interface_connectivity!(elements, mpi_interfaces, mesh)

  return mpi_interfaces
end


function start_mpi_receive!(dg::Dg2D)
  for (index, d) in enumerate(dg.mpi_neighbor_domain_ids)
    dg.mpi_recv_requests[index] = MPI.Irecv!(dg.mpi_recv_buffers[index], d, d, mpi_comm())
  end
end


# Initialize connectivity between elements and interfaces
function init_mpi_interface_connectivity!(elements, mpi_interfaces, mesh::TreeMesh{2})
  # Reset interface count
  count = 0

  # Iterate over all elements to find neighbors and to connect via mpi_interfaces
  for element_id in 1:nelements(elements)
    # Get cell id
    cell_id = elements.cell_ids[element_id]

    # Loop over directions
    for direction in 1:n_directions(mesh.tree)
      # If no neighbor exists, current cell is small and thus we need a mortar
      if !has_neighbor(mesh.tree, cell_id, direction)
        continue
      end

      # Skip if neighbor has children
      neighbor_cell_id = mesh.tree.neighbor_ids[direction, cell_id]
      if has_children(mesh.tree, neighbor_cell_id)
        continue
      end

      # Skip if neighbor is on this domain -> create regular interface instead
      if is_parallel() && is_own_cell(mesh.tree, neighbor_cell_id)
        continue
      end

      # Create interface between elements
      count += 1
      mpi_interfaces.local_element_ids[count] = element_id

      if direction in (2, 4) # element is "left" of interface, remote cell is "right" of interface
        mpi_interfaces.remote_sides[count] = 2
      else
        mpi_interfaces.remote_sides[count] = 1
      end

      # Set orientation (x -> 1, y -> 2)
      if direction in (1, 2) # x-direction
        mpi_interfaces.orientations[count] = 1
      else # y-direction
        mpi_interfaces.orientations[count] = 2
      end
    end
  end

  @assert count == nmpiinterfaces(mpi_interfaces) ("Actual interface count ($count) does not match "
                                                   * "expectations $(nmpiinterfaces(mpi_interfaces))")
end


# Initialize connectivity between MPI neighbor domains
function init_mpi_neighbor_connectivity(elements, mpi_interfaces, mesh::TreeMesh{2})
  tree = mesh.tree

  # Determine neighbor domains and sides for MPI interfaces
  neighbor_domain_ids = fill(-1, nmpiinterfaces(mpi_interfaces))
  # The global interface id is the smaller of the (globally unique) neighbor cell ids, multiplied by
  # number of directions (2 * ndims) plus direction minus one
  global_interface_ids = fill(-1, nmpiinterfaces(mpi_interfaces))
  my_domain_id = domain_id()
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
    neighbor_domain_ids[interface_id] = tree.domain_ids[remote_cell_id]
    if local_cell_id < remote_cell_id
      global_interface_ids[interface_id] = 2 * ndims(tree) * local_cell_id + direction - 1
    else
      global_interface_ids[interface_id] = (2 * ndims(tree) * remote_cell_id +
                                            opposite_direction(direction) - 1)
    end
  end

  # Get sorted, unique neighbor domain ids
  mpi_neighbor_domain_ids = unique(sort(neighbor_domain_ids))

  # Sort interfaces by global interface id
  p = sortperm(global_interface_ids)
  neighbor_domain_ids .= neighbor_domain_ids[p]
  interface_ids = collect(1:nmpiinterfaces(mpi_interfaces))[p]

  # For each neighbor domain id, init connectivity data structures
  mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_domain_ids))
  for (index, d) in enumerate(mpi_neighbor_domain_ids)
    mpi_neighbor_interfaces[index] = interface_ids[findall(x->(x == d), neighbor_domain_ids)]
  end

  # Sanity check that we counted all interfaces exactly once
  @assert sum(length(v) for v in mpi_neighbor_interfaces) == nmpiinterfaces(mpi_interfaces)

  return mpi_neighbor_domain_ids, mpi_neighbor_interfaces
end


# Initialize MPI data structures
function init_mpi_data_structures(mpi_neighbor_interfaces, ::Val{NDIMS}, ::Val{NVARS},
                                  ::Val{POLYDEG}) where {NDIMS, NVARS, POLYDEG}
  data_size = NVARS * (POLYDEG + 1)^(NDIMS - 1)
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


function prolong2mpiinterfaces!(dg::Dg2D)
  equation = equations(dg)

  Threads.@threads for s in 1:dg.n_mpi_interfaces
    local_element_id = dg.mpi_interfaces.local_element_ids[s]
    if dg.mpi_interfaces.orientations[s] == 1 # interface in x-direction
      if dg.mpi_interfaces.remote_sides[s] == 1 # local element in positive direction
        for j in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.mpi_interfaces.u[2, v, j, s] = dg.elements.u[v,          1, j, local_element_id]
        end
      else # local element in negative direction
        for j in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.mpi_interfaces.u[1, v, j, s] = dg.elements.u[v, nnodes(dg), j, local_element_id]
        end
      end
    else # interface in y-direction
      if dg.mpi_interfaces.remote_sides[s] == 1 # local element in positive direction
        for i in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.mpi_interfaces.u[2, v, i, s] = dg.elements.u[v, i,          1, local_element_id]
        end
      else # local element in negative direction
        for i in 1:nnodes(dg), v in 1:nvariables(dg)
          dg.mpi_interfaces.u[1, v, i, s] = dg.elements.u[v, i, nnodes(dg), local_element_id]
        end
      end
    end
  end
end


function start_mpi_send!(dg::Dg2D)
  data_size = nvariables(dg) * nnodes(dg)^(ndims(dg) - 1)

  for d in 1:length(dg.mpi_neighbor_domain_ids)
    send_buffer = dg.mpi_send_buffers[d]

    for (index, s) in enumerate(dg.mpi_neighbor_interfaces[d])
      first = (index - 1) * data_size + 1
      last =  (index - 1) * data_size + data_size

      if dg.mpi_interfaces.remote_sides[s] == 1 # local element in positive direction
        @views send_buffer[first:last] .= vec(dg.mpi_interfaces.u[2, :, :, s])
      else # local element in negative direction
        @views send_buffer[first:last] .= vec(dg.mpi_interfaces.u[1, :, :, s])
      end
    end
  end

  # Start sending
  for (index, d) in enumerate(dg.mpi_neighbor_domain_ids)
    dg.mpi_send_requests[index] = MPI.Isend(dg.mpi_send_buffers[index], d, domain_id(), mpi_comm())
  end
end


function finish_mpi_receive!(dg::Dg2D)
  data_size = nvariables(dg) * nnodes(dg)^(ndims(dg) - 1)

  # Start receiving and unpack received data until all communication is finished
  d, _ = MPI.Waitany!(dg.mpi_recv_requests)
  while d != 0
    recv_buffer = dg.mpi_recv_buffers[d]

    for (index, s) in enumerate(dg.mpi_neighbor_interfaces[d])
      first = (index - 1) * data_size + 1
      last =  (index - 1) * data_size + data_size

      if dg.mpi_interfaces.remote_sides[s] == 1 # local element in positive direction
        @views vec(dg.mpi_interfaces.u[1, :, :, s]) .= recv_buffer[first:last]
      else # local element in negative direction
        @views vec(dg.mpi_interfaces.u[2, :, :, s]) .= recv_buffer[first:last]
      end
    end

    d, _ = MPI.Waitany!(dg.mpi_recv_requests)
  end
end


# Calculate and store the surface fluxes (standard Riemann and nonconservative parts) at an MPI interface
# OBS! Regarding the nonconservative terms: 1) currently only needed for the MHD equations
#                                           2) not implemented for MPI
calc_mpi_interface_flux!(dg::Dg2D) = calc_mpi_interface_flux!(dg.elements.surface_flux_values,
                                                              have_nonconservative_terms(dg.equations),
                                                              dg)

function calc_mpi_interface_flux!(surface_flux_values, nonconservative_terms::Val{false}, dg::Dg2D)
  @unpack surface_flux_function = dg
  @unpack u, local_element_ids, orientations, remote_sides = dg.mpi_interfaces

  Threads.@threads for s in 1:dg.n_mpi_interfaces
    # Get local neighboring element
    element_id = local_element_ids[s]

    # Determine interface direction with respect to element:
    if orientations[s] == 1 # interface in x-direction
      if remote_sides[s] == 1 # local element in positive direction
        direction = 1
      else # local element in negative direction
        direction = 2
      end
    else # interface in y-direction
      if remote_sides[s] == 1 # local element in positive direction
        direction = 3
      else # local element in negative direction
        direction = 4
      end
    end

    for i in 1:nnodes(dg)
      # Call pointwise Riemann solver
      u_ll, u_rr = get_surface_node_vars(u, dg, i, s)
      flux = surface_flux_function(u_ll, u_rr, orientations[s], equations(dg))

      # Copy flux to local element storage
      for v in 1:nvariables(dg)
        surface_flux_values[v, i, direction, element_id] = flux[v]
      end
    end
  end
end


function finish_mpi_send!(dg::Dg2D)
  MPI.Waitall!(dg.mpi_send_requests)
end
