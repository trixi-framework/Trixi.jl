module Interpolation

using GaussQuadrature: legendre, both

export interpolate_nodes
export calcdhat
export polynomialderivativematrix
export polynomialinterpolationmatrix
export barycentricweights
export calclhat
export lagrangeinterpolatingpolynomials
export gausslobatto


# Interpolate data using the given Vandermonde matrix and return interpolated values.
function interpolate_nodes(data_in::AbstractArray{T, 2},
                           vandermonde::AbstractArray{T, 2}, nvars_::Integer) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in = size(vandermonde, 2)
  data_out = zeros(eltype(data_in), nvars_, n_nodes_out)

  for i = 1:n_nodes_out
    for j = 1:n_nodes_in
      for v = 1:nvars_
        data_out[v, i] += vandermonde[i, j] * data_in[v, j]
      end
    end
  end

  return data_out
end


# Calculate the Dhat matrix
function calcdhat(nodes, weights)
  n_nodes = length(nodes)
  dhat = polynomialderivativematrix(nodes)
  dhat = transpose(dhat)

  for n = 1:n_nodes, j = 1:n_nodes
    dhat[j, n] *= -weights[n] / weights[j]
  end

  return dhat
end


# Calculate the polynomial derivative matrix D
function polynomialderivativematrix(nodes)
  n_nodes = length(nodes)
  d = zeros(n_nodes, n_nodes)
  wbary = barycentricweights(nodes)

  for i = 1:n_nodes, j = 1:n_nodes
    if j != i
      d[i, j] = wbary[j] / wbary[i] * 1 / (nodes[i] - nodes[j])
      d[i, i] -= d[i, j]
    end
  end

  return d
end


# Calculate and interpolation matrix (Vandermonde matrix) between two given sets of nodes
function polynomialinterpolationmatrix(nodes_in, nodes_out)
  n_nodes_in = length(nodes_in)
  n_nodes_out = length(nodes_out)
  wbary_in = barycentricweights(nodes_in)
  vdm = zeros(n_nodes_out, n_nodes_in)

  for k = 1:n_nodes_out
    match = false
    for j = 1:n_nodes_in
      if isapprox(nodes_out[k], nodes_in[j], rtol=eps())
        match = true
        vdm[k, j] = 1
      end
    end

    if match == false
      s = 0.0
      for j = 1:n_nodes_in
        t = wbary_in[j] / (nodes_out[k] - nodes_in[j])
        vdm[k, j] = t
        s += t
      end
      for j = 1:n_nodes_in
        vdm[k, j] = vdm[k, j] / s
      end
    end
  end

  return vdm
end


# Calculate the barycentric weights for a given node distribution.
function barycentricweights(nodes)
  n_nodes = length(nodes)
  weights = ones(n_nodes)

  for j = 2:n_nodes, k = 1:(j-1)
    weights[k] *= nodes[k] - nodes[j]
    weights[j] *= nodes[j] - nodes[k]
  end

  for j = 1:n_nodes
    weights[j] = 1 / weights[j]
  end

  return weights
end


# Calculate Lhat.
function calclhat(x::Float64, nodes, weights)
  n_nodes = length(nodes)
  wbary = barycentricweights(nodes)

  lhat = lagrangeinterpolatingpolynomials(x, nodes, wbary)

  for i = 1:n_nodes
    lhat[i] /= weights[i]
  end

  return lhat
end


# Calculate Lagrange polynomials for a given node distribution.
function lagrangeinterpolatingpolynomials(x::Float64, nodes, wbary)
  n_nodes = length(nodes)
  polynomials = zeros(n_nodes)

  for i = 1:n_nodes
    if isapprox(x, nodes[i], rtol=eps(x))
      polynomials[i] = 1
      return polynomials
    end
  end

  for i = 1:n_nodes
    polynomials[i] = wbary[i] / (x - nodes[i])
  end
  total = sum(polynomials)

  for i = 1:n_nodes
    polynomials[i] /= total
  end

  return polynomials
end


# Calculate nodes and weights for Legendre-Gauss-Lobatto quadratue.
function gausslobatto(n_nodes::Integer)
  return legendre(n_nodes, both)
end


end
