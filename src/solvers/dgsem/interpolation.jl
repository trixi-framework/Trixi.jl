
# Naive implementations of multiply_dimensionwise used to demonstrate the functionality
# without performance optimizations and for testing correctness of the optimized versions
# implemented below.
function multiply_dimensionwise_naive(matrix::AbstractMatrix,
                                      data_in::AbstractArray{<:Any, 2})
    size_out = size(matrix, 1)
    size_in = size(matrix, 2)
    n_vars = size(data_in, 1)
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

function multiply_dimensionwise_naive(matrix::AbstractMatrix,
                                      data_in::AbstractArray{<:Any, 3})
    size_out = size(matrix, 1)
    size_in = size(matrix, 2)
    n_vars = size(data_in, 1)
    data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out,
                     size_out)

    for j in 1:size_out, i in 1:size_out
        for jj in 1:size_in, ii in 1:size_in
            for v in 1:n_vars
                data_out[v, i, j] += matrix[i, ii] * matrix[j, jj] * data_in[v, ii, jj]
            end
        end
    end

    return data_out
end

function multiply_dimensionwise_naive(matrix::AbstractMatrix,
                                      data_in::AbstractArray{<:Any, 4})
    size_out = size(matrix, 1)
    size_in = size(matrix, 2)
    n_vars = size(data_in, 1)
    data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out,
                     size_out, size_out)

    for k in 1:size_out, j in 1:size_out, i in 1:size_out
        for kk in 1:size_in, jj in 1:size_in, ii in 1:size_in
            for v in 1:n_vars
                data_out[v, i, j, k] += matrix[i, ii] * matrix[j, jj] * matrix[k, kk] *
                                        data_in[v, ii, jj, kk]
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
    n_vars = size(data_in, 1)
    data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out)

    multiply_dimensionwise!(data_out, matrix, data_in)

    return data_out
end

function multiply_dimensionwise(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 3})
    # 2D
    # optimized version of multiply_dimensionwise_naive
    size_out = size(matrix, 1)
    n_vars = size(data_in, 1)
    data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out,
                     size_out)

    multiply_dimensionwise!(data_out, matrix, data_in)

    return data_out
end

function multiply_dimensionwise(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 4})
    # 3D
    # optimized version of multiply_dimensionwise_naive
    size_out = size(matrix, 1)
    n_vars = size(data_in, 1)
    data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out,
                     size_out, size_out)

    multiply_dimensionwise!(data_out, matrix, data_in)

    return data_out
end

# In the following, there are several optimized in-place versions of multiply_dimensionwise.
# These may make use of advanced optimization features such as the macro `@tullio` from Tullio.jl,
# which basically uses an Einstein summation convention syntax.
# Another possibility is `@turbo` from LoopVectorization.jl. The runtime performance could be even
# optimized further by using `@turbo inline=true for` instead of `@turbo for`, but that comes at the
# cost of increased latency, at least on some systems...

# 1D version
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 2}, matrix::AbstractMatrix,
                                 data_in::AbstractArray{<:Any, 2})
    # @tullio threads=false data_out[v, i] = matrix[i, ii] * data_in[v, ii]
    @turbo for i in axes(data_out, 2), v in axes(data_out, 1)
        res = zero(eltype(data_out))
        for ii in axes(matrix, 2)
            res += matrix[i, ii] * data_in[v, ii]
        end
        data_out[v, i] = res
    end

    return nothing
end

# 1D version for scalars
# Instead of having a leading dimension of size 1 in `data_out, data_in`, this leading dimension
# of size unity is dropped, resulting in one dimension less than in `multiply_dimensionwise!`.
function multiply_scalar_dimensionwise!(data_out::AbstractArray{<:Any, 1},
                                        matrix::AbstractMatrix,
                                        data_in::AbstractArray{<:Any, 1})
    # @tullio threads=false data_out[i] = matrix[i, ii] * data_in[ii]
    @turbo for i in axes(data_out, 1)
        res = zero(eltype(data_out))
        for ii in axes(matrix, 2)
            res += matrix[i, ii] * data_in[ii]
        end
        data_out[i] = res
    end

    return nothing
end

# 1D version, apply matrixJ to data_inJ
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 2}, matrix1::AbstractMatrix,
                                 data_in1::AbstractArray{<:Any, 2}, matrix2::AbstractMatrix,
                                 data_in2::AbstractArray{<:Any, 2})
    # @tullio threads=false data_out[v, i] = matrix1[i, ii] * data_in1[v, ii] + matrix2[i, ii] * data_in2[v, ii]
    # TODO: LoopVectorization upgrade
    #   We would like to use `@turbo` for the outermost loop possibly fuse both inner
    #   loops, but that does currently not work because of limitations of
    #   LoopVectorizationjl. However, Chris Elrod is planning to address this in
    #   the future, cf. https://github.com/JuliaSIMD/LoopVectorization.jl/issues/230#issuecomment-810632972
    @turbo for i in axes(data_out, 2), v in axes(data_out, 1)
        res = zero(eltype(data_out))
        for ii in axes(matrix1, 2)
            res += matrix1[i, ii] * data_in1[v, ii]
        end
        data_out[v, i] = res
    end
    @turbo for i in axes(data_out, 2), v in axes(data_out, 1)
        res = zero(eltype(data_out))
        for ii in axes(matrix2, 2)
            res += matrix2[i, ii] * data_in2[v, ii]
        end
        data_out[v, i] += res
    end

    return nothing
end

# 2D version
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 3}, matrix::AbstractMatrix,
                                 data_in::AbstractArray{<:Any, 3},
                                 tmp1 = zeros(eltype(data_out), size(data_out, 1),
                                              size(matrix, 1), size(matrix, 2)))

    # Interpolate in x-direction
    # @tullio threads=false tmp1[v, i, j]     = matrix[i, ii] * data_in[v, ii, j]
    @turbo for j in axes(tmp1, 3), i in axes(tmp1, 2), v in axes(tmp1, 1)
        res = zero(eltype(tmp1))
        for ii in axes(matrix, 2)
            res += matrix[i, ii] * data_in[v, ii, j]
        end
        tmp1[v, i, j] = res
    end

    # Interpolate in y-direction
    # @tullio threads=false data_out[v, i, j] = matrix[j, jj] * tmp1[v, i, jj]
    @turbo for j in axes(data_out, 3), i in axes(data_out, 2), v in axes(data_out, 1)
        res = zero(eltype(data_out))
        for jj in axes(matrix, 2)
            res += matrix[j, jj] * tmp1[v, i, jj]
        end
        data_out[v, i, j] = res
    end

    return nothing
end

# 2D version for scalars
# Instead of having a leading dimension of size 1 in `data_out, data_in`, this leading dimension
# of size unity is dropped, resulting in one dimension less than in `multiply_dimensionwise!`.
function multiply_scalar_dimensionwise!(data_out::AbstractArray{<:Any, 2},
                                        matrix::AbstractMatrix,
                                        data_in::AbstractArray{<:Any, 2},
                                        tmp1 = zeros(eltype(data_out), size(matrix, 1),
                                                     size(matrix, 2)))

    # Interpolate in x-direction
    # @tullio threads=false     tmp1[i, j] = matrix[i, ii] * data_in[ii, j]
    @turbo for j in axes(tmp1, 2), i in axes(tmp1, 1)
        res = zero(eltype(tmp1))
        for ii in axes(matrix, 2)
            res += matrix[i, ii] * data_in[ii, j]
        end
        tmp1[i, j] = res
    end

    # Interpolate in y-direction
    # @tullio threads=false data_out[i, j] = matrix[j, jj] * tmp1[i, jj]
    @turbo for j in axes(data_out, 2), i in axes(data_out, 1)
        res = zero(eltype(data_out))
        for jj in axes(matrix, 2)
            res += matrix[j, jj] * tmp1[i, jj]
        end
        data_out[i, j] = res
    end

    return nothing
end

# 2D version, apply matrixJ to dimension J of data_in
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 3},
                                 matrix1::AbstractMatrix, matrix2::AbstractMatrix,
                                 data_in::AbstractArray{<:Any, 3},
                                 tmp1 = zeros(eltype(data_out), size(data_out, 1),
                                              size(matrix1, 1), size(matrix1, 2)))

    # Interpolate in x-direction
    # @tullio threads=false tmp1[v, i, j]     = matrix1[i, ii] * data_in[v, ii, j]
    @turbo for j in axes(tmp1, 3), i in axes(tmp1, 2), v in axes(tmp1, 1)
        res = zero(eltype(tmp1))
        for ii in axes(matrix1, 2)
            res += matrix1[i, ii] * data_in[v, ii, j]
        end
        tmp1[v, i, j] = res
    end

    # Interpolate in y-direction
    # @tullio threads=false data_out[v, i, j] = matrix2[j, jj] * tmp1[v, i, jj]
    @turbo for j in axes(data_out, 3), i in axes(data_out, 2), v in axes(data_out, 1)
        res = zero(eltype(data_out))
        for jj in axes(matrix2, 2)
            res += matrix2[j, jj] * tmp1[v, i, jj]
        end
        data_out[v, i, j] = res
    end

    return nothing
end

# 2D version, apply matrixJ to dimension J of data_in and add the result to data_out
function add_multiply_dimensionwise!(data_out::AbstractArray{<:Any, 3},
                                     matrix1::AbstractMatrix, matrix2::AbstractMatrix,
                                     data_in::AbstractArray{<:Any, 3},
                                     tmp1 = zeros(eltype(data_out), size(data_out, 1),
                                                  size(matrix1, 1), size(matrix1, 2)))

    # Interpolate in x-direction
    # @tullio threads=false tmp1[v, i, j]     = matrix1[i, ii] * data_in[v, ii, j]
    @turbo for j in axes(tmp1, 3), i in axes(tmp1, 2), v in axes(tmp1, 1)
        res = zero(eltype(tmp1))
        for ii in axes(matrix1, 2)
            res += matrix1[i, ii] * data_in[v, ii, j]
        end
        tmp1[v, i, j] = res
    end

    # Interpolate in y-direction
    # @tullio threads=false data_out[v, i, j] += matrix2[j, jj] * tmp1[v, i, jj]
    @turbo for j in axes(data_out, 3), i in axes(data_out, 2), v in axes(data_out, 1)
        res = zero(eltype(data_out))
        for jj in axes(matrix2, 2)
            res += matrix2[j, jj] * tmp1[v, i, jj]
        end
        data_out[v, i, j] += res
    end

    return nothing
end

# 3D version
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 4}, matrix::AbstractMatrix,
                                 data_in::AbstractArray{<:Any, 4},
                                 tmp1 = zeros(eltype(data_out), size(data_out, 1),
                                              size(matrix, 1), size(matrix, 2),
                                              size(matrix, 2)),
                                 tmp2 = zeros(eltype(data_out), size(data_out, 1),
                                              size(matrix, 1), size(matrix, 1),
                                              size(matrix, 2)))

    # Interpolate in x-direction
    # @tullio threads=false tmp1[v, i, j, k]     = matrix[i, ii] * data_in[v, ii, j, k]
    @turbo for k in axes(tmp1, 4), j in axes(tmp1, 3), i in axes(tmp1, 2),
               v in axes(tmp1, 1)

        res = zero(eltype(tmp1))
        for ii in axes(matrix, 2)
            res += matrix[i, ii] * data_in[v, ii, j, k]
        end
        tmp1[v, i, j, k] = res
    end

    # Interpolate in y-direction
    # @tullio threads=false tmp2[v, i, j, k]     = matrix[j, jj] * tmp1[v, i, jj, k]
    @turbo for k in axes(tmp2, 4), j in axes(tmp2, 3), i in axes(tmp2, 2),
               v in axes(tmp2, 1)

        res = zero(eltype(tmp2))
        for jj in axes(matrix, 2)
            res += matrix[j, jj] * tmp1[v, i, jj, k]
        end
        tmp2[v, i, j, k] = res
    end

    # Interpolate in z-direction
    # @tullio threads=false data_out[v, i, j, k] = matrix[k, kk] * tmp2[v, i, j, kk]
    @turbo for k in axes(data_out, 4), j in axes(data_out, 3), i in axes(data_out, 2),
               v in axes(data_out, 1)

        res = zero(eltype(data_out))
        for kk in axes(matrix, 2)
            res += matrix[k, kk] * tmp2[v, i, j, kk]
        end
        data_out[v, i, j, k] = res
    end

    return nothing
end

# 3D version for scalars
# Instead of having a leading dimension of size 1 in `data_out, data_in`, this leading dimension
# of size unity is dropped, resulting in one dimension less than in `multiply_dimensionwise!`.
function multiply_scalar_dimensionwise!(data_out::AbstractArray{<:Any, 3},
                                        matrix::AbstractMatrix,
                                        data_in::AbstractArray{<:Any, 3},
                                        tmp1 = zeros(eltype(data_out), size(matrix, 1),
                                                     size(matrix, 2), size(matrix, 2)),
                                        tmp2 = zeros(eltype(data_out), size(matrix, 1),
                                                     size(matrix, 1), size(matrix, 2)))

    # Interpolate in x-direction
    # @tullio threads=false tmp1[i, j, k]     = matrix[i, ii] * data_in[ii, j, k]
    @turbo for k in axes(tmp1, 3), j in axes(tmp1, 2), i in axes(tmp1, 1)
        res = zero(eltype(tmp1))
        for ii in axes(matrix, 2)
            res += matrix[i, ii] * data_in[ii, j, k]
        end
        tmp1[i, j, k] = res
    end

    # Interpolate in y-direction
    # @tullio threads=false tmp2[i, j, k]     = matrix[j, jj] * tmp1[i, jj, k]
    @turbo for k in axes(tmp2, 3), j in axes(tmp2, 2), i in axes(tmp2, 1)
        res = zero(eltype(tmp2))
        for jj in axes(matrix, 2)
            res += matrix[j, jj] * tmp1[i, jj, k]
        end
        tmp2[i, j, k] = res
    end

    # Interpolate in z-direction
    # @tullio threads=false data_out[i, j, k] = matrix[k, kk] * tmp2[i, j, kk]
    @turbo for k in axes(data_out, 3), j in axes(data_out, 2), i in axes(data_out, 1)
        res = zero(eltype(data_out))
        for kk in axes(matrix, 2)
            res += matrix[k, kk] * tmp2[i, j, kk]
        end
        data_out[i, j, k] = res
    end

    return nothing
end

# 3D version, apply matrixJ to dimension J of data_in
function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 4},
                                 matrix1::AbstractMatrix, matrix2::AbstractMatrix,
                                 matrix3::AbstractMatrix,
                                 data_in::AbstractArray{<:Any, 4},
                                 tmp1 = zeros(eltype(data_out), size(data_out, 1),
                                              size(matrix1, 1), size(matrix1, 2),
                                              size(matrix1, 2)),
                                 tmp2 = zeros(eltype(data_out), size(data_out, 1),
                                              size(matrix1, 1), size(matrix1, 1),
                                              size(matrix1, 2)))

    # Interpolate in x-direction
    # @tullio threads=false tmp1[v, i, j, k]     = matrix1[i, ii] * data_in[v, ii, j, k]
    @turbo for k in axes(tmp1, 4), j in axes(tmp1, 3), i in axes(tmp1, 2),
               v in axes(tmp1, 1)

        res = zero(eltype(tmp1))
        for ii in axes(matrix1, 2)
            res += matrix1[i, ii] * data_in[v, ii, j, k]
        end
        tmp1[v, i, j, k] = res
    end

    # Interpolate in y-direction
    # @tullio threads=false tmp2[v, i, j, k]     = matrix2[j, jj] * tmp1[v, i, jj, k]
    @turbo for k in axes(tmp2, 4), j in axes(tmp2, 3), i in axes(tmp2, 2),
               v in axes(tmp2, 1)

        res = zero(eltype(tmp1))
        for jj in axes(matrix2, 2)
            res += matrix2[j, jj] * tmp1[v, i, jj, k]
        end
        tmp2[v, i, j, k] = res
    end

    # Interpolate in z-direction
    # @tullio threads=false data_out[v, i, j, k] = matrix3[k, kk] * tmp2[v, i, j, kk]
    @turbo for k in axes(data_out, 4), j in axes(data_out, 3), i in axes(data_out, 2),
               v in axes(data_out, 1)

        res = zero(eltype(data_out))
        for kk in axes(matrix3, 2)
            res += matrix3[k, kk] * tmp2[v, i, j, kk]
        end
        data_out[v, i, j, k] = res
    end

    return nothing
end

# 3D version, apply matrixJ to dimension J of data_in and add the result to data_out
function add_multiply_dimensionwise!(data_out::AbstractArray{<:Any, 4},
                                     matrix1::AbstractMatrix, matrix2::AbstractMatrix,
                                     matrix3::AbstractMatrix,
                                     data_in::AbstractArray{<:Any, 4},
                                     tmp1 = zeros(eltype(data_out), size(data_out, 1),
                                                  size(matrix1, 1), size(matrix1, 2),
                                                  size(matrix1, 2)),
                                     tmp2 = zeros(eltype(data_out), size(data_out, 1),
                                                  size(matrix1, 1), size(matrix1, 1),
                                                  size(matrix1, 2)))

    # Interpolate in x-direction
    # @tullio threads=false tmp1[v, i, j, k]     = matrix1[i, ii] * data_in[v, ii, j, k]
    @turbo for k in axes(tmp1, 4), j in axes(tmp1, 3), i in axes(tmp1, 2),
               v in axes(tmp1, 1)

        res = zero(eltype(tmp1))
        for ii in axes(matrix1, 2)
            res += matrix1[i, ii] * data_in[v, ii, j, k]
        end
        tmp1[v, i, j, k] = res
    end

    # Interpolate in y-direction
    # @tullio threads=false tmp2[v, i, j, k]     = matrix2[j, jj] * tmp1[v, i, jj, k]
    @turbo for k in axes(tmp2, 4), j in axes(tmp2, 3), i in axes(tmp2, 2),
               v in axes(tmp2, 1)

        res = zero(eltype(tmp1))
        for jj in axes(matrix2, 2)
            res += matrix2[j, jj] * tmp1[v, i, jj, k]
        end
        tmp2[v, i, j, k] = res
    end

    # Interpolate in z-direction
    # @tullio threads=false data_out[v, i, j, k] += matrix3[k, kk] * tmp2[v, i, j, kk]
    @turbo for k in axes(data_out, 4), j in axes(data_out, 3), i in axes(data_out, 2),
               v in axes(data_out, 1)

        res = zero(eltype(data_out))
        for kk in axes(matrix3, 2)
            res += matrix3[k, kk] * tmp2[v, i, j, kk]
        end
        data_out[v, i, j, k] += res
    end

    return nothing
end
