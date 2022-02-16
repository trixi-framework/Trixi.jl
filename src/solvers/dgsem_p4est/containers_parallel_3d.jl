# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# Initialize node_indices of MPI interface container
@inline function init_mpi_interface_node_indices!(mpi_interfaces::P4estMPIInterfaceContainer{3},
                                                  faces, local_side, orientation,
                                                  mpi_interface_id)
  # Align interface at the primary element (primary element has surface indices (:i_forward, :j_forward)).
  # The secondary element needs to be indexed differently.
  if local_side == 1
    surface_index1 = :i_forward
    surface_index2 = :j_forward
  else # local_side == 2
    surface_index1, surface_index2 = orientation_to_indices_p4est(faces[2], faces[1], orientation)
  end

  if faces[local_side] == 0
    # Index face in negative x-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (:begin, surface_index1, surface_index2)
  elseif faces[local_side] == 1
    # Index face in positive x-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (:end, surface_index1, surface_index2)
  elseif faces[local_side] == 2
    # Index face in negative y-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (surface_index1, :begin, surface_index2)
  elseif faces[local_side] == 3
    # Index face in positive y-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (surface_index1, :end, surface_index2)
  elseif faces[local_side] == 4
    # Index face in negative z-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (surface_index1, surface_index2, :begin)
  else # faces[local_side] == 5
    # Index face in positive z-direction
    mpi_interfaces.node_indices[mpi_interface_id] = (surface_index1, surface_index2, :end)
  end

  return mpi_interfaces
end


end # muladd