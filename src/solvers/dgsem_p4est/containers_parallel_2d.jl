# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# Initialize node_indices of MPI interface container
@inline function init_mpi_interface_node_indices!(mpi_interfaces::P4estMPIInterfaceContainer{2},
                                                  local_face, local_side, orientation,
                                                  mpi_interface_id)
  # Align interface in positive coordinate direction of primary element.
  # For orientation == 1, the secondary element needs to be indexed backwards
  # relative to the interface.
  if local_side == 1 || orientation == 0
    # Forward indexing
    i = :i_forward
  else
    # Backward indexing
    i = :i_backward
  end

  if local_face == 0
    # Index face in negative x-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (:begin, i)
  elseif local_face == 1
    # Index face in positive x-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (:end, i)
  elseif local_face == 2
    # Index face in negative y-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (i, :begin)
  else # local_face == 3
    # Index face in positive y-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (i, :end)
  end

  return mpi_interfaces
end


end # muladd