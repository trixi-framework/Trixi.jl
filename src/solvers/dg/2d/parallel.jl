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
    mpi_recv_requests[index] = MPI.Irecv!(dg.mpi_recv_buffers[index], d, d, mpi_comm())
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
      mpi_interfaces.orientations[count] = div(direction, 2)
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
  my_domain_id = domain_id()
  for interface_id in 1:nmpiinterfaces(mpi_interfaces)
    orientation = mpi_interfaces.orientations[interface_id]
    remote_side = mpi_interfaces.remote_sides[interface_id]
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
  end

  # Get sorted, unique neighbor domain ids
  mpi_neighbor_domain_ids = unique(sort(neighbor_domain_ids))

  # For each neighbor domain id, init connectivity data structures
  mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_domain_ids))
  for (index, d) in enumerate(mpi_neighbor_domain_ids)
    mpi_neighbor_interfaces[index] = findall(x->(x == d), neighbor_domain_ids)
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
end


function start_mpi_send!(dg::Dg2D)
  error("pack buffers")
  for (index, d) in enumerate(dg.mpi_neighbor_domain_ids)
    mpi_send_requests[index] = MPI.Isend(dg.mpi_send_buffers[index], d, domain_id(), mpi_comm())
  end
end


function finish_mpi_receive!(dg::Dg2D)
end


function calc_mpi_interface_flux!(dg::Dg2D)
end


function finish_mpi_send!(dg::Dg2D)
  MPI.Waitall!(dg.mpi_send_requests)
end
