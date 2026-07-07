# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_boundary_node_coordinates!(boundaries, element, count, direction,
                                         elements, mesh::TreeMesh2D,
                                         basis::UniformFiniteVolumeBasis)
    el_node_coords = elements.node_coordinates
    bnd_node_coords = boundaries.node_coordinates
    cell_id = elements.cell_ids[element]
    cell_center_x = mesh.tree.coordinates[1, cell_id]
    cell_center_y = mesh.tree.coordinates[2, cell_id]
    # because inverse_jacobian = 2/dx and dx=dy
    half_cell_length = inv(elements.inverse_jacobian[element])

    x_left = cell_center_x - half_cell_length
    x_right = cell_center_x + half_cell_length
    y_lower = cell_center_y - half_cell_length
    y_upper = cell_center_y + half_cell_length

    if direction == 1 # -x direction
        bnd_node_coords[1, :, count] .= x_left
        @views bnd_node_coords[2, :, count] .= el_node_coords[2, 1, :, element]
    elseif direction == 2 # +x direction
        bnd_node_coords[1, :, count] .= x_right
        @views bnd_node_coords[2, :, count] .= el_node_coords[2, end, :, element]
    elseif direction == 3 # -y direction
        bnd_node_coords[2, :, count] .= y_lower
        @views bnd_node_coords[1, :, count] .= el_node_coords[1, :, 1, element]
    elseif direction == 4 # +y direction
        bnd_node_coords[2, :, count] .= y_upper
        @views bnd_node_coords[1, :, count] .= el_node_coords[1, :, end, element]
    else
        error("should not happen")
    end

    return nothing
end
end # @muladd
