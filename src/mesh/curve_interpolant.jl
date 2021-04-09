
struct GammaCurve{RealT<:Real, NNODES}
  nodes        ::SVector{NNODES, RealT}
  bary_weights ::SVector{NNODES, RealT}
  x_vals       ::SVector{NNODES, RealT}
  y_vals       ::SVector{NNODES, RealT}
end


# construct a single instance of the gamma curve struct
function GammaCurve(RealT, curve_polydeg, curve_x_vals, curve_y_vals)
  nnodes_ = curve_polydeg + 1

  nodes, _ = chebyshev_gauss_lobatto_nodes_weights(nnodes_)
  wbary    = barycentric_weights(nodes)

  return GammaCurve{RealT, nnodes_}(nodes, wbary, curve_x_vals, curve_y_vals)
end


# evalute the Gamma curve interpolant at a particular point s and return the (x,y) coordinate
function evaluate_at(s, bndy_curve::GammaCurve)

   @unpack nodes, bary_weights, x_vals, y_vals = bndy_curve

   x_point = lagrange_interpolation(s, nodes, x_vals, bary_weights)
   y_point = lagrange_interpolation(s, nodes, y_vals, bary_weights)

   return x_point, y_point
end


# evalute the derivative of a Gamma curve interpolant at a particular point s
# and return the (x,y) coordinate
function derivative_at(s, bndy_curve::GammaCurve)

   @unpack nodes, bary_weights, x_vals, y_vals = bndy_curve

   x_point_prime = lagrange_interpolation_derivative(s, nodes, x_vals, bary_weights)
   y_point_prime = lagrange_interpolation_derivative(s, nodes, y_vals, bary_weights)

   return x_point_prime, y_point_prime
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
  n_nodes = length(nodes)
  numerator   = 0
  denominator = 0

  for j in 1:n_nodes
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
  n_nodes   = length(nodes)
  at_node   = false
  numerator = 0
  i         = 0

  for j in 1:n_nodes
    if isapprox(x, nodes[j], rtol=eps(x))
      at_node     = true
      p           = fvals[j]
      denominator = -wbary[j]
      i           = j
    end
  end

  if at_node
    for j in 1:n_nodes
      if j != i
        numerator += wbary[j] * ( p - fvals[j] ) / ( x - nodes[j] )
      end
    end
  else
    denominator = 0
    p = lagrange_interpolation(x, nodes, fvals, wbary)
    for j in 1:n_nodes
      t            = wbary[j] / (x - nodes[j])
      numerator   += t * ( p - fvals[j] ) / ( x - nodes[j] )
      denominator += t
    end
  end

  return numerator/denominator # p_prime
end
