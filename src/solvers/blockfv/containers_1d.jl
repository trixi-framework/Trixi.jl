# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_boundary_node_coordinates!(boundaries, element, count, direction,
                                         elements, mesh::TreeMesh1D,
                                         basis::UniformFiniteVolumeBasis)
    bnd_node_coords = boundaries.node_coordinates
    cell_id = elements.cell_ids[element]
    cell_center = mesh.tree.coordinates[1, cell_id]
    half_cell_length = inv(elements.inverse_jacobian[element]) # because inverse_jacobian = 2/dx

    orientation = 1 # always 1 in 1D
    if direction == 1
        bnd_node_coords[orientation, count] = cell_center - half_cell_length
    elseif direction == 2
        bnd_node_coords[orientation, count] = cell_center + half_cell_length
    else
        error("should not happen")
    end

    return nothing
end
end # @muladd
