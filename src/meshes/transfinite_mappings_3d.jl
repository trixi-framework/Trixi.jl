# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# Illustration of the corner (circled), edge (braces), and face index numbering convention
# used in these functions.
#
#                    ⑧────────────────────────{7}────────────────────────⑦
#                   ╱│                                                   ╱│
#                  ╱ │                                                  ╱ │
#                 ╱  │                                                 ╱  │
#                ╱   │                5 (+z)                          ╱   │
#               ╱    │                                               ╱    │
#              ╱     │                                              ╱     │
#            {12}    │                                            {11}    │
#            ╱       │                                            ╱       │
#           ╱        │                                           ╱        │
#          ╱         │                    2 (+y)                ╱         │
#         ╱          │                                         ╱          │
#        ╱          {8}                                       ╱          {6}
#       ╱            │                                       ╱            │
#      ⑤─────────────────────────{3}───────────────────────⑥    4 (+x)   │
#      │             │                                      │             │
#      │             │                                      │             │
#      │             │                                      │             │
#      │    6 (-x)   │                                      │             │
#      │             │                                      │             │
#      │             │                                      │             │
#      │             │                                      │             │
#      │             │                                      │             │
#      │             ④────────────────────────{5}──────────│─────────────③
#      │            ╱                                       │            ╱
#      │           ╱              1 (-y)                    │           ╱
#     {4}         ╱                                        {2}         ╱
#      │         ╱                                          │         ╱
#      │        ╱                                           │        ╱
#      │      {9}                                           │      {10}
#      │      ╱                                             │      ╱
#      │     ╱                                              │     ╱  Global coordinates:
#      │    ╱                                               │    ╱       z
#      │   ╱                             3 (-z)             │   ╱        ↑   y
#      │  ╱                                                 │  ╱         │  ╱
#      │ ╱                                                  │ ╱          │ ╱
#      │╱                                                   │╱           └─────> x
#      ①───────────────────────{1}─────────────────────────②


# Transfinite mapping formula from a point (xi, eta, zeta) in reference space [-1,1]^3 to a
# physical coordinate (x, y, z) for a hexahedral element with straight sides
function straight_side_hex_map(xi, eta, zeta, corner_points)

  coordinate = zeros(eltype(xi), 3)
  for j in 1:3
    coordinate[j] += (0.125 * ( corner_points[j, 1] * (1 - xi) * (1 - eta) * (1 - zeta)
                              + corner_points[j, 2] * (1 + xi) * (1 - eta) * (1 - zeta)
                              + corner_points[j, 3] * (1 + xi) * (1 + eta) * (1 - zeta)
                              + corner_points[j, 4] * (1 - xi) * (1 + eta) * (1 - zeta)
                              + corner_points[j, 5] * (1 - xi) * (1 - eta) * (1 + zeta)
                              + corner_points[j, 6] * (1 + xi) * (1 - eta) * (1 + zeta)
                              + corner_points[j, 7] * (1 + xi) * (1 + eta) * (1 + zeta)
                              + corner_points[j, 8] * (1 - xi) * (1 + eta) * (1 + zeta) ) )
  end

  return coordinate
end


# Construct the (x, y, z) node coordinates in the volume of a straight sided hexahedral element
function calc_node_coordinates!(node_coordinates::AbstractArray{<:Any, 5}, element, nodes, corners)

  for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
    node_coordinates[:, i, j, k, element] .= straight_side_hex_map(nodes[i], nodes[j], nodes[k], corners)
  end

  return node_coordinates
end


# Transfinite mapping formula from a point (xi, eta, zeta) in reference space [-1,1]^3 to a point
# (x,y,z) in physical coordinate space for a hexahedral element with general curved sides
# See Section 4.3
# - Andrew R. Winters (2014)
#   Discontinuous Galerkin spectral element approximations for the reflection and
#   transmission of waves from moving material interfaces
#   [PhD thesis, Florida State University](https://diginole.lib.fsu.edu/islandora/object/fsu%3A185342)
function transfinite_hex_map(xi, eta, zeta, face_curves::AbstractVector{<:CurvedFace})

  coordinate = zeros(eltype(xi), 3)
  face_values = zeros(eltype(xi), (3, 6))
  edge_values = zeros(eltype(xi), (3, 12))
  corners = zeros(eltype(xi), (3, 8))

  # Compute values along the face edges
  edge_values[:, 1] .= evaluate_at(SVector(xi,   -1), face_curves[1])
  edge_values[:, 2] .= evaluate_at(SVector( 1, zeta), face_curves[1])
  edge_values[:, 3] .= evaluate_at(SVector(xi,    1), face_curves[1])
  edge_values[:, 4] .= evaluate_at(SVector(-1, zeta), face_curves[1])

  edge_values[:, 5] .= evaluate_at(SVector(xi,   -1), face_curves[2])
  edge_values[:, 6] .= evaluate_at(SVector( 1, zeta), face_curves[2])
  edge_values[:, 7] .= evaluate_at(SVector(xi,    1), face_curves[2])
  edge_values[:, 8] .= evaluate_at(SVector(-1, zeta), face_curves[2])

  edge_values[:, 9]  .= evaluate_at(SVector(eta, -1), face_curves[6])
  edge_values[:, 10] .= evaluate_at(SVector(eta, -1), face_curves[4])
  edge_values[:, 11] .= evaluate_at(SVector(eta,  1), face_curves[4])
  edge_values[:, 12] .= evaluate_at(SVector(eta,  1), face_curves[6])

  # Compute values on the face
  face_values[:, 1] .= evaluate_at(SVector( xi, zeta), face_curves[1])
  face_values[:, 2] .= evaluate_at(SVector( xi, zeta), face_curves[2])
  face_values[:, 3] .= evaluate_at(SVector( xi,  eta), face_curves[3])
  face_values[:, 4] .= evaluate_at(SVector(eta, zeta), face_curves[4])
  face_values[:, 5] .= evaluate_at(SVector( xi,  eta), face_curves[5])
  face_values[:, 6] .= evaluate_at(SVector(eta, zeta), face_curves[6])

  # Pull the eight corner values and compute the straight sided hex mapping
  corners[:,1] .= face_curves[1].coordinates[:, 1,   1]
  corners[:,2] .= face_curves[1].coordinates[:, end, 1]
  corners[:,3] .= face_curves[2].coordinates[:, end, 1]
  corners[:,4] .= face_curves[2].coordinates[:, 1,   1]
  corners[:,5] .= face_curves[1].coordinates[:, 1,   end]
  corners[:,6] .= face_curves[1].coordinates[:, end, end]
  corners[:,7] .= face_curves[2].coordinates[:, end, end]
  corners[:,8] .= face_curves[2].coordinates[:, 1,   end]

  coordinate_straight = straight_side_hex_map(xi, eta, zeta, corners)

  # Compute the transfinite mapping
  for j in 1:3
    # Linear interpolation between opposite faces
    coordinate[j] = ( 0.5 * ( face_values[j, 6] * (1 - xi  ) + face_values[j, 4] * (1 + xi  )
                            + face_values[j, 1] * (1 - eta ) + face_values[j, 2] * (1 + eta )
                            + face_values[j, 3] * (1 - zeta) + face_values[j, 5] * (1 + zeta) ) )

    # Edge corrections to ensure faces match
    coordinate[j] -= ( 0.25 * ( edge_values[j, 1 ] * (1 - eta) * (1 - zeta)
                              + edge_values[j, 2 ] * (1 + xi ) * (1 - eta )
                              + edge_values[j, 3 ] * (1 - eta) * (1 + zeta)
                              + edge_values[j, 4 ] * (1 - xi ) * (1 - eta )
                              + edge_values[j, 5 ] * (1 + eta) * (1 - zeta)
                              + edge_values[j, 6 ] * (1 + xi ) * (1 + eta )
                              + edge_values[j, 7 ] * (1 + eta) * (1 + zeta)
                              + edge_values[j, 8 ] * (1 - xi ) * (1 + eta )
                              + edge_values[j, 9 ] * (1 - xi ) * (1 - zeta)
                              + edge_values[j, 10] * (1 + xi ) * (1 - zeta)
                              + edge_values[j, 11] * (1 + xi ) * (1 + zeta)
                              + edge_values[j, 12] * (1 - xi ) * (1 + zeta) ) )

    # Subtracted interior twice, so add back the straight-sided hexahedral mapping
    coordinate[j] += coordinate_straight[j]
  end

  return coordinate
end


# Construct the (x, y, z) node coordinates in the volume of a curved sided hexahedral element
function calc_node_coordinates!(node_coordinates::AbstractArray{<:Any, 5}, element, nodes,
                                face_curves::AbstractVector{<:CurvedFace})

  for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
    node_coordinates[:, i, j, k, element] .= transfinite_hex_map(nodes[i], nodes[j], nodes[k], face_curves)
  end

  return node_coordinates
end


end # @muladd
