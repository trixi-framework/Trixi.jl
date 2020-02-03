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


function interpolate_nodes(data_in::AbstractArray{T, 2},
                           vandermonde::AbstractArray{T, 2}, nvars_::Integer) where T
  nnodes_out = size(vandermonde, 1)
  nnodes_in = size(vandermonde, 2)
  data_out = zeros(eltype(data_in), nvars_, nnodes_out)

  for i = 1:nnodes_out
    for j = 1:nnodes_in
      for v = 1:nvars_
        data_out[v, i] += vandermonde[i, j] * data_in[v, j]
      end
    end
  end

  return data_out
end


function calcdhat(nodes, weights)
  nnodes = length(nodes)
  dhat = polynomialderivativematrix(nodes)
  dhat = transpose(dhat)

  for n = 1:nnodes, j = 1:nnodes
    dhat[j, n] *= -weights[n] / weights[j]
  end

  return dhat
end


function polynomialderivativematrix(nodes)
  nnodes = length(nodes)
  d = zeros(nnodes, nnodes)
  wbary = barycentricweights(nodes)

  for i = 1:nnodes, j = 1:nnodes
    if j != i
      d[i, j] = wbary[j] / wbary[i] * 1 / (nodes[i] - nodes[j])
      d[i, i] -= d[i, j]
    end
  end

  return d
end


function polynomialinterpolationmatrix(nodes_in, nodes_out)
  nnodes_in = length(nodes_in)
  nnodes_out = length(nodes_out)
  wbary_in = barycentricweights(nodes_in)
  vdm = zeros(nnodes_out, nnodes_in)

  for k = 1:nnodes_out
    match = false
    for j = 1:nnodes_in
      if isapprox(nodes_out[k], nodes_in[j], rtol=eps())
        match = true
        vdm[k, j] = 1
      end
    end

    if match == false
      s = 0.0
      for j = 1:nnodes_in
        t = wbary_in[j] / (nodes_out[k] - nodes_in[j])
        vdm[k, j] = t
        s += t
      end
      for j = 1:nnodes_in
        vdm[k, j] = vdm[k, j] / s
      end
    end
  end

  return vdm
end


function barycentricweights(nodes)
  nnodes = length(nodes)
  weights = ones(nnodes)

  for j = 2:nnodes, k = 1:(j-1)
    weights[k] *= nodes[k] - nodes[j]
    weights[j] *= nodes[j] - nodes[k]
  end

  for j = 1:nnodes
    weights[j] = 1 / weights[j]
  end

  return weights
end


function calclhat(x::Float64, nodes, weights)
  nnodes = length(nodes)
  wbary = barycentricweights(nodes)

  lhat = lagrangeinterpolatingpolynomials(x, nodes, wbary)

  for i = 1:nnodes
    lhat[i] /= weights[i]
  end

  return lhat
end


function lagrangeinterpolatingpolynomials(x::Float64, nodes, wbary)
  nnodes = length(nodes)
  polynomials = zeros(nnodes)

  for i = 1:nnodes
    if isapprox(x, nodes[i], rtol=eps(x))
      polynomials[i] = 1
      return polynomials
    end
  end

  for i = 1:nnodes
    polynomials[i] = wbary[i] / (x - nodes[i])
  end
  total = sum(polynomials)

  for i = 1:nnodes
    polynomials[i] /= total
  end

  return polynomials
end


function gausslobatto(nnodes::Integer)
  return legendre(nnodes, both)
end


end
