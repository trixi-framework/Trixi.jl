# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


#     CurvedSurface{RealT<:Real}
#
# Contains the data needed to represent a curve with data points (x,y) as a Lagrange polynomial
# interpolant written in barycentric form at a given set of nodes.
struct CurvedSurface{RealT<:Real}
  nodes               ::Vector{RealT}
  barycentric_weights ::Vector{RealT}
  coordinates         ::Array{RealT, 2} #[nnodes, ndims]
end


# evaluate the Gamma curve interpolant at a particular point s and return the (x,y) coordinate
function evaluate_at(s, boundary_curve::CurvedSurface)

   @unpack nodes, barycentric_weights, coordinates = boundary_curve

   x_coordinate_at_s_on_boundary_curve = lagrange_interpolation(s, nodes, view(coordinates, :, 1),
                                                                barycentric_weights)
   y_coordinate_at_s_on_boundary_curve = lagrange_interpolation(s, nodes, view(coordinates, :, 2),
                                                                barycentric_weights)

   return x_coordinate_at_s_on_boundary_curve, y_coordinate_at_s_on_boundary_curve
end


# evaluate the derivative of a Gamma curve interpolant at a particular point s
# and return the (x,y) coordinate
function derivative_at(s, boundary_curve::CurvedSurface)

   @unpack nodes, barycentric_weights, coordinates = boundary_curve

   x_coordinate_at_s_on_boundary_curve_prime = lagrange_interpolation_derivative(s, nodes,
                                                                                 view(coordinates, :, 1),
                                                                                 barycentric_weights)
   y_coordinate_at_s_on_boundary_curve_prime = lagrange_interpolation_derivative(s, nodes,
                                                                                 view(coordinates, :, 2),
                                                                                 barycentric_weights)
   return x_coordinate_at_s_on_boundary_curve_prime, y_coordinate_at_s_on_boundary_curve_prime
end


# Chebysehv-Gauss-Lobatto nodes and weights for use with curved boundaries
function chebyshev_gauss_lobatto_nodes_weights(n_nodes::Integer)

  # Initialize output
  nodes   = zeros(n_nodes)
  weights = zeros(n_nodes)

  # Get polynomial degree for convenience
  N = n_nodes - 1

  for j in 1:n_nodes
    nodes[j]   = -cospi( (j-1) / N )
    weights[j] = pi / N
  end
  weights[1]   = 0.5 * weights[1]
  weights[end] = 0.5 * weights[end]

  return nodes, weights
end


# Calculate Lagrange interpolating polynomial of a function f(x) at a given point x for a given
# node distribution.
function lagrange_interpolation(x, nodes, fvals, wbary)
# Barycentric two formulation of Lagrange interpolant
  numerator   = zero(eltype(fvals))
  denominator = zero(eltype(fvals))

  for j in eachindex(nodes)
    if isapprox(x, nodes[j], rtol=eps(x))
      return fvals[j]
    end
    t            = wbary[j] / ( x - nodes[j] )
    numerator   += t * fvals[j]
    denominator += t
  end

  return numerator/denominator
end


# Calculate derivative of a Lagrange interpolating polynomial of a function f(x) at a given
# point x for a given node distribution.
function lagrange_interpolation_derivative(x, nodes, fvals, wbary)

  at_node   = false
  numerator = zero(eltype(fvals))
  i         = 0

  for j in eachindex(nodes)
    if isapprox(x, nodes[j])
      at_node     = true
      p           = fvals[j]
      denominator = -wbary[j]
      i           = j
    end
  end

  if at_node
    for j in eachindex(nodes)
      if j != i
        numerator += wbary[j] * ( p - fvals[j] ) / ( x - nodes[j] )
      end
    end
  else
    denominator = zero(eltype(fvals))
    p = lagrange_interpolation(x, nodes, fvals, wbary)
    for j in eachindex(nodes)
      t            = wbary[j] / (x - nodes[j])
      numerator   += t * ( p - fvals[j] ) / ( x - nodes[j] )
      denominator += t
    end
  end

  return numerator/denominator # p_prime
end


end # @muladd
