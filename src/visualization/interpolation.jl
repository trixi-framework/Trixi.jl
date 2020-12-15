
# Interpolate data using the given Vandermonde matrix and return interpolated values (1D version).
function interpolate_nodes(data_in::AbstractArray{T, 2},
                           vandermonde::AbstractArray{T, 2}, n_vars::Integer) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in = size(vandermonde, 2)
  data_out = zeros(eltype(data_in), n_vars, n_nodes_out)

  for i = 1:n_nodes_out
    for ii = 1:n_nodes_in
      for v = 1:n_vars
        data_out[v, i] += vandermonde[i, ii] * data_in[v, ii]
      end
    end
  end

  return data_out
end


# Interpolate data using the given Vandermonde matrix and return interpolated values (2D version).
function interpolate_nodes(data_in::AbstractArray{T, 3},
                           vandermonde, n_vars) where T
  n_nodes_out = size(vandermonde, 1)
  data_out = zeros(eltype(data_in), n_vars, n_nodes_out, n_nodes_out)
  interpolate_nodes!(data_out, data_in, vandermonde, n_vars)
end

function interpolate_nodes!(data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3},
                            vandermonde, n_vars) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  for j in 1:n_nodes_out
    for i in 1:n_nodes_out
      for v in 1:n_vars
        acc = zero(eltype(data_out))
        for jj in 1:n_nodes_in
          for ii in 1:n_nodes_in
            acc += vandermonde[i, ii] * data_in[v, ii, jj] * vandermonde[j, jj]
          end
        end
        data_out[v, i, j] = acc
      end
    end
  end

  return data_out
end


# Interpolate data using the given Vandermonde matrix and return interpolated values (3D version).
function interpolate_nodes(data_in::AbstractArray{T, 4},
                           vandermonde, n_vars) where T
  n_nodes_out = size(vandermonde, 1)
  data_out = zeros(eltype(data_in), n_vars, n_nodes_out, n_nodes_out, n_nodes_out)
  interpolate_nodes!(data_out, data_in, vandermonde, n_vars)
end

function interpolate_nodes!(data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4},
                            vandermonde, n_vars) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_vars
      acc = zero(eltype(data_out))
      for kk in 1:n_nodes_in, jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * vandermonde[k, kk] * data_in[v, ii, jj, kk]
      end
      data_out[v, i, j, k] = acc
    end
  end

  return data_out
end
