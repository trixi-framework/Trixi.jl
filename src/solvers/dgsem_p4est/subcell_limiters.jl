# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

###############################################################################
# Auxiliary routine `get_boundary_outer_state` for non-periodic domains

"""
    get_boundary_outer_state(u_inner, t,
                             boundary_condition::BoundaryConditionDirichlet,
                             normal_direction
                             mesh, equations, dg, cache, indices...)
For subcell limiting, the calculation of local bounds for non-periodic domains requires the boundary
outer state. This function returns the boundary value  for [`BoundaryConditionDirichlet`](@ref) at
time `t` and for node with spatial indices `indices` at the boundary with `normal_direction`.

Should be used together with [`P4estMesh`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function get_boundary_outer_state(u_inner, t,
                                          boundary_condition::BoundaryConditionDirichlet,
                                          normal_direction,
                                          mesh::P4estMesh,
                                          equations, dg, cache, indices...)
    (; node_coordinates) = cache.elements

    x = get_node_coords(node_coordinates, equations, dg, indices...)
    u_outer = boundary_condition.boundary_value_function(x, t, equations)

    return u_outer
end

@inline function get_boundary_outer_state(u_inner, t,
                                          boundary_condition::BoundaryConditionCharacteristic,
                                          normal_direction,
                                          mesh::P4estMesh, equations, dg, cache,
                                          indices...)
    (; node_coordinates) = cache.elements

    x = get_node_coords(node_coordinates, equations, dg, indices...)
    u_outer = boundary_condition.boundary_value_function(boundary_condition.outer_boundary_value_function,
                                                         u_inner,
                                                         normal_direction,
                                                         x, t, equations)

    return u_outer
end
end # @muladd
