
# Naive implementations of multiply_dimensionwise used to demonstrate the functionality
# without performance optimizations and for testing correctness of the optimized versions
# implemented below.
function multiply_dimensionwise_naive(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 2})
  size_out = size(matrix, 1)
  size_in  = size(matrix, 2)
  n_vars   = size(data_in, 1)
  data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out)

  for i in 1:size_out
    for ii in 1:size_in
      for v in 1:n_vars
        data_out[v, i] += matrix[i, ii] * data_in[v, ii]
      end
    end
  end

  return data_out
end

function multiply_dimensionwise_naive(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 3})
  size_out = size(matrix, 1)
  size_in  = size(matrix, 2)
  n_vars   = size(data_in, 1)
  data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out, size_out)

  for j in 1:size_out, i in 1:size_out
    for jj in 1:size_in, ii in 1:size_in
      for v in 1:n_vars
        data_out[v, i, j] += matrix[i, ii] * matrix[j, jj] * data_in[v, ii, jj]
      end
    end
  end

  return data_out
end

function multiply_dimensionwise_naive(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 4})
  size_out = size(matrix, 1)
  size_in  = size(matrix, 2)
  n_vars   = size(data_in, 1)
  data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out, size_out, size_out)

  for k in 1:size_out, j in 1:size_out, i in 1:size_out
    for kk in 1:size_in, jj in 1:size_in, ii in 1:size_in
      for v in 1:n_vars
        data_out[v, i, j, k] += matrix[i, ii] * matrix[j, jj] * matrix[k, kk] * data_in[v, ii, jj, kk]
      end
    end
  end

  return data_out
end

"""
    multiply_dimensionwise(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, NDIMS+1})

Multiply the array `data_in` by `matrix` in each coordinate direction, where `data_in`
is assumed to have the first coordinate for the number of variables and the remaining coordinates
are multiplied by `matrix`.
"""
function multiply_dimensionwise(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 2})
  # 1D
  # optimized version of multiply_dimensionwise_naive
  size_out = size(matrix, 1)
  n_vars   = size(data_in, 1)
  data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out)

  multiply_dimensionwise!(data_out, matrix, data_in)

  return data_out
end

function multiply_dimensionwise(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 3})
  # 2D
  # optimized version of multiply_dimensionwise_naive
  size_out = size(matrix, 1)
  n_vars   = size(data_in, 1)
  data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out, size_out)

  multiply_dimensionwise!(data_out, matrix, data_in)

  return data_out
end

function multiply_dimensionwise(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 4})
  # 3D
  # optimized version of multiply_dimensionwise_naive
  size_out = size(matrix, 1)
  n_vars   = size(data_in, 1)
  data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out, size_out, size_out)

  multiply_dimensionwise!(data_out, matrix, data_in)

  return data_out
end


# In the following, there are several optimized in-place versions of multiply_dimensionwise.
# These make use of the macro `@tullio` from Tullio.jl, which basically uses an Einstein
# summation convention syntax.

# 1D version
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 2}, matrix::AbstractMatrix,
                                 data_in ::AbstractArray{<:Any, 2})
  @tullio threads=false data_out[v, i] = matrix[i, ii] * data_in[v, ii]

  return nothing
end

# 1D version for scalars
function multiply_scalar_dimensionwise!(data_out::AbstractArray{<:Any, 1},
                                        matrix::AbstractMatrix,
                                        data_in ::AbstractArray{<:Any, 1})
  @tullio threads=false data_out[i] = matrix[i, ii] * data_in[ii]

  return nothing
end

# 1D version, apply matrixJ to data_inJ
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 2}, matrix1::AbstractMatrix,
                                 data_in1::AbstractArray{<:Any, 2}, matrix2::AbstractMatrix,
                                 data_in2::AbstractArray{<:Any, 2})
  @tullio threads=false data_out[v, i] = matrix1[i, ii] * data_in1[v, ii] + matrix2[i, ii] * data_in2[v, ii]

  return nothing
end

# 2D version
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 3}, matrix::AbstractMatrix,
                                 data_in:: AbstractArray{<:Any, 3},
                                 tmp1=zeros(eltype(data_out), size(data_out, 1), size(matrix, 1), size(matrix, 2)))

  # Interpolate in x-direction
  @tullio threads=false tmp1[v, i, j]     = matrix[i, ii] * data_in[v, ii, j]

  # Interpolate in y-direction
  @tullio threads=false data_out[v, i, j] = matrix[j, jj] * tmp1[v, i, jj]

  return nothing
end

# 2D version for scalars
function multiply_scalar_dimensionwise!(data_out::AbstractArray{<:Any, 2},
                                        matrix::AbstractMatrix,
                                        data_in:: AbstractArray{<:Any, 2},
                                        tmp1=zeros(eltype(data_out), size(matrix, 1), size(matrix, 2)))

  # Interpolate in x-direction
  @tullio threads=false     tmp1[i, j] = matrix[i, ii] * data_in[ii, j]

  # Interpolate in y-direction
  @tullio threads=false data_out[i, j] = matrix[j, jj] * tmp1[i, jj]

  return nothing
end

# 2D version, apply matrixJ to dimension J of data_in
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 3},
                                 matrix1::AbstractMatrix, matrix2::AbstractMatrix,
                                 data_in:: AbstractArray{<:Any, 3},
                                 tmp1=zeros(eltype(data_out), size(data_out, 1), size(matrix1, 1), size(matrix1, 2)))

  # Interpolate in x-direction
  @tullio threads=false tmp1[v, i, j]     = matrix1[i, ii] * data_in[v, ii, j]

  # Interpolate in y-direction
  @tullio threads=false data_out[v, i, j] = matrix2[j, jj] * tmp1[v, i, jj]

  return nothing
end

# 2D version, apply matrixJ to dimension J of data_in and add the result to data_out
function add_multiply_dimensionwise!(data_out::AbstractArray{<:Any, 3},
                                     matrix1::AbstractMatrix, matrix2::AbstractMatrix,
                                     data_in:: AbstractArray{<:Any, 3},
                                     tmp1=zeros(eltype(data_out), size(data_out, 1), size(matrix1, 1), size(matrix1, 2)))

  # Interpolate in x-direction
  @tullio threads=false tmp1[v, i, j]     = matrix1[i, ii] * data_in[v, ii, j]

  # Interpolate in y-direction
  @tullio threads=false data_out[v, i, j] += matrix2[j, jj] * tmp1[v, i, jj]

  return nothing
end

# 3D version
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 4}, matrix::AbstractMatrix,
                                 data_in:: AbstractArray{<:Any, 4},
                                 tmp1=zeros(eltype(data_out), size(data_out, 1), size(matrix, 1), size(matrix, 2), size(matrix, 2)),
                                 tmp2=zeros(eltype(data_out), size(data_out, 1), size(matrix, 1), size(matrix, 1), size(matrix, 2)))

  # Interpolate in x-direction
  @tullio threads=false tmp1[v, i, j, k]     = matrix[i, ii] * data_in[v, ii, j, k]

  # Interpolate in y-direction
  @tullio threads=false tmp2[v, i, j, k]     = matrix[j, jj] * tmp1[v, i, jj, k]

  # Interpolate in z-direction
  @tullio threads=false data_out[v, i, j, k] = matrix[k, kk] * tmp2[v, i, j, kk]

  return nothing
end

# 3D version, apply matrixJ to dimension J of data_in
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 4},
                                 matrix1::AbstractMatrix, matrix2::AbstractMatrix, matrix3::AbstractMatrix,
                                 data_in:: AbstractArray{<:Any, 4},
                                 tmp1=zeros(eltype(data_out), size(data_out, 1), size(matrix1, 1), size(matrix1, 2), size(matrix1, 2)),
                                 tmp2=zeros(eltype(data_out), size(data_out, 1), size(matrix1, 1), size(matrix1, 1), size(matrix1, 2)))

  # Interpolate in x-direction
  @tullio threads=false tmp1[v, i, j, k]     = matrix1[i, ii] * data_in[v, ii, j, k]

  # Interpolate in y-direction
  @tullio threads=false tmp2[v, i, j, k]     = matrix2[j, jj] * tmp1[v, i, jj, k]

  # Interpolate in z-direction
  @tullio threads=false data_out[v, i, j, k] = matrix3[k, kk] * tmp2[v, i, j, kk]

  return nothing
end

# 3D version, apply matrixJ to dimension J of data_in and add the result to data_out
function add_multiply_dimensionwise!(data_out::AbstractArray{<:Any, 4},
                                     matrix1::AbstractMatrix, matrix2::AbstractMatrix, matrix3::AbstractMatrix,
                                     data_in:: AbstractArray{<:Any, 4},
                                     tmp1=zeros(eltype(data_out), size(data_out, 1), size(matrix1, 1), size(matrix1, 2), size(matrix1, 2)),
                                     tmp2=zeros(eltype(data_out), size(data_out, 1), size(matrix1, 1), size(matrix1, 1), size(matrix1, 2)))

  # Interpolate in x-direction
  @tullio threads=false tmp1[v, i, j, k]     = matrix1[i, ii] * data_in[v, ii, j, k]

  # Interpolate in y-direction
  @tullio threads=false tmp2[v, i, j, k]     = matrix2[j, jj] * tmp1[v, i, jj, k]

  # Interpolate in z-direction
  @tullio threads=false data_out[v, i, j, k] += matrix3[k, kk] * tmp2[v, i, j, kk]

  return nothing
end

