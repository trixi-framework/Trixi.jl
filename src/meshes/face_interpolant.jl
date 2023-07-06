# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

#     CurvedFace{RealT<:Real}
#
# Contains the data needed to represent a curved face with data points (x,y,z) as a Lagrange polynomial
# interpolant written in barycentric form at a given set of nodes.
struct CurvedFace{RealT <: Real}
    nodes::Vector{RealT}
    barycentric_weights::Vector{RealT}
    coordinates::Array{RealT, 3} #[ndims, nnodes, nnodes]
end

# evaluate the Gamma face interpolant at a particular point s = (s_1, s_2) and return the (x,y,z) coordinate
function evaluate_at(s, boundary_face::CurvedFace)
    @unpack nodes, barycentric_weights, coordinates = boundary_face

    x_coordinate_at_s_on_boundary_face = lagrange_interpolation_2d(s, nodes,
                                                                   view(coordinates, 1,
                                                                        :, :),
                                                                   barycentric_weights)
    y_coordinate_at_s_on_boundary_face = lagrange_interpolation_2d(s, nodes,
                                                                   view(coordinates, 2,
                                                                        :, :),
                                                                   barycentric_weights)
    z_coordinate_at_s_on_boundary_face = lagrange_interpolation_2d(s, nodes,
                                                                   view(coordinates, 3,
                                                                        :, :),
                                                                   barycentric_weights)

    return x_coordinate_at_s_on_boundary_face,
           y_coordinate_at_s_on_boundary_face,
           z_coordinate_at_s_on_boundary_face
end

# Calculate a 2D Lagrange interpolating polynomial in barycentric 2 form
# of a function f(x,y) at a given coordinate (x,y) for a given node distribution.
function lagrange_interpolation_2d(x, nodes, function_values, barycentric_weights)
    f_intermediate = zeros(eltype(function_values), length(nodes))
    for j in eachindex(nodes)
        f_intermediate[j] = lagrange_interpolation(x[2], nodes,
                                                   view(function_values, j, :),
                                                   barycentric_weights)
    end
    point_value = lagrange_interpolation(x[1], nodes, f_intermediate,
                                         barycentric_weights)

    return point_value
end
end # @muladd
