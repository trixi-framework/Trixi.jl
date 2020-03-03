module L2Mortar

using ..Interpolation: gauss_lobatto_nodes_weights, barycentric_weights,
                       lagrange_interpolating_polynomials


function calc_forward_upper(n_nodes)
  # Calculate nodes, weights, and barycentric weights
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  wbary = barycentric_weights(nodes)

  # Calculate projection matrix (actually: interpolation)
  operator = zeros(n_nodes, n_nodes)
  for j in 1:n_nodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] + 1), nodes, wbary)
    for i in 1:n_nodes
      operator[j, i] = poly[i]
    end
  end

  return operator
end


function calc_forward_lower(n_nodes)
  # Calculate nodes, weights, and barycentric weights
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  wbary = barycentric_weights(nodes)

  # Calculate projection matrix (actually: interpolation)
  operator = zeros(n_nodes, n_nodes)
  for j in 1:n_nodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] - 1), nodes, wbary)
    for i in 1:n_nodes
      operator[j, i] = poly[i]
    end
  end

  return operator
end


function calc_reverse_upper(n_nodes)
  # Calculate nodes, weights, and barycentric weights
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  wbary = barycentric_weights(nodes)

  # Calculate projection matrix (actually: discrete L2 projection with errors)
  operator = zeros(n_nodes, n_nodes)
  for j in 1:n_nodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] + 1), nodes, wbary)
    for i in 1:n_nodes
      operator[i, j] = 1/2 * poly[i] * weights[j]/weights[i]
    end
  end

  return operator
end


function calc_reverse_lower(n_nodes)
  # Calculate nodes, weights, and barycentric weights
  nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
  wbary = barycentric_weights(nodes)

  # Calculate projection matrix (actually: discrete L2 projection with errors)
  operator = zeros(n_nodes, n_nodes)
  for j in 1:n_nodes
    poly = lagrange_interpolating_polynomials(1/2 * (nodes[j] - 1), nodes, wbary)
    for i in 1:n_nodes
      operator[i, j] = 1/2 * poly[i] * weights[j]/weights[i]
    end
  end

  return operator
end


end # module Mortar
