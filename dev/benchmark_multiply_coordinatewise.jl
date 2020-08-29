import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using BenchmarkTools
using BSON
using LoopVectorization
using MuladdMacro
using StaticArrays

###################################################################################################
# 2D versions
function multiply_coordinatewise_sequential!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  tmp = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in)
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp[v, i, j] += vandermonde[i, ii] * data_in[v, ii, j]
      end
    end
  end

  # Interpolate in y-direction
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j] += vandermonde[j, jj] * tmp[v, i, jj]
      end
    end
  end

  return data_out
end

@muladd function multiply_coordinatewise_sequential_muladd!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  tmp = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in)
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp[v, i, j] += vandermonde[i, ii] * data_in[v, ii, j]
      end
    end
  end

  # Interpolate in y-direction
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j] += vandermonde[j, jj] * tmp[v, i, jj]
      end
    end
  end

  return data_out
end

function multiply_coordinatewise_sequential_simd!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  tmp = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in)
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    @simd for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp[v, i, j] += vandermonde[i, ii] * data_in[v, ii, j]
      end
    end
  end

  # Interpolate in y-direction
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    @simd for jj in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j] += vandermonde[j, jj] * tmp[v, i, jj]
      end
    end
  end

  return data_out
end

@muladd function multiply_coordinatewise_sequential_simd_muladd!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  tmp = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in)
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    @simd for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp[v, i, j] += vandermonde[i, ii] * data_in[v, ii, j]
      end
    end
  end

  # Interpolate in y-direction
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    @simd for jj in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j] += vandermonde[j, jj] * tmp[v, i, jj]
      end
    end
  end

  return data_out
end

function multiply_coordinatewise_sequential_avx!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables;
    tmp=zeros(eltype(data_out), n_variables, size(vandermonde, 1), size(vandermonde, 2))) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  @avx for j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp[v, i, j] += vandermonde[i, ii] * data_in[v, ii, j]
      end
    end
  end

  # Interpolate in y-direction
  @avx for j in 1:n_nodes_in, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j] += vandermonde[j, jj] * tmp[v, i, jj]
      end
    end
  end

  return data_out
end


function multiply_coordinatewise_squeezed!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_variables
      acc = zero(eltype(data_out))
      for jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii]*  vandermonde[j, jj] * data_in[v, ii, jj]
      end
      data_out[v, i, j] = acc
    end
  end

  return data_out
end

@muladd function multiply_coordinatewise_squeezed_muladd!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_variables
      acc = zero(eltype(data_out))
      for jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * data_in[v, ii, jj]
      end
      data_out[v, i, j] = acc
    end
  end

  return data_out
end

function multiply_coordinatewise_squeezed_simd!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
    @simd for v in 1:n_variables
      acc = zero(eltype(data_out))
      for jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * data_in[v, ii, jj]
      end
      data_out[v, i, j] = acc
    end
  end

  return data_out
end

@muladd function multiply_coordinatewise_squeezed_simd_muladd!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
    @simd for v in 1:n_variables
      acc = zero(eltype(data_out))
      for jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * data_in[v, ii, jj]
      end
      data_out[v, i, j] = acc
    end
  end

  return data_out
end

function multiply_coordinatewise_squeezed_avx!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @avx for j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_variables
      acc = zero(eltype(data_out))
      for jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * data_in[v, ii, jj]
      end
      data_out[v, i, j] = acc
    end
  end

  return data_out
end


function run_benchmarks_2d(;n_variables_total=4, n_variables_interp=n_variables_total, n_nodes_in=4, n_nodes_out=2*n_nodes_in)
  data_in  = randn(n_variables_total, n_nodes_in,  n_nodes_in)
  data_out = randn(n_variables_total, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde_static  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)

  println("multiply_coordinatewise_sequential!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_sequential!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("vandermonde_static")
  display(@benchmark multiply_coordinatewise_sequential!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_sequential_muladd!")
  display(@benchmark multiply_coordinatewise_sequential_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_sequential_simd!")
  display(@benchmark multiply_coordinatewise_sequential_simd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_sequential_simd_muladd!")
  display(@benchmark multiply_coordinatewise_sequential_simd_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_sequential_avx!")
  display(@benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed!")
  display(@benchmark multiply_coordinatewise_squeezed!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed_muladd!")
  display(@benchmark multiply_coordinatewise_squeezed_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed_simd!")
  display(@benchmark multiply_coordinatewise_squeezed_simd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed_simd_muladd!")
  display(@benchmark multiply_coordinatewise_squeezed_simd_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed_avx!")
  display(@benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  nothing
end

# TODO
run_benchmarks_2d(n_variables_total=4, n_variables_interp=4, n_nodes_in=4, n_nodes_out=4)


function compute_benchmarks_2d(;n_variables_total=4, n_variables_interp=n_variables_total, n_nodes_in=4, n_nodes_out=2*n_nodes_in)
  data_in  = randn(n_variables_total, n_nodes_in,  n_nodes_in)
  data_out = randn(n_variables_interp, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)
  tmp = zeros(eltype(data_out), n_variables_interp, n_nodes_out, n_nodes_in)

  println("n_variables_total = ", n_variables_total, "; n_variables_interp = ", n_variables_interp,
          "; n_nodes_in = ", n_nodes_in, "; n_nodes_out = ", n_nodes_out)
  sequential = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde), $(n_variables_interp))
  sequential_prealloc = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde), $(n_variables_interp), tmp=$(tmp))
  squeezed = @benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde), $(n_variables_interp))

  return time(median(sequential)), time(median(sequential_prealloc)), time(median(squeezed))
end

function compute_benchmarks_2d(n_variables_total_list, n_nodes_in_list)
  sequential          = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_prealloc = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  squeezed            = zeros(length(n_variables_total_list), length(n_nodes_in_list))

  # n_variables_interp = n_variables_total, n_nodes_out = n_nodes_in
  # mortar
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = n_variables_total
      n_nodes_out = n_nodes_in
      sequential[idx_variables, idx_nodes], sequential_prealloc[idx_variables, idx_nodes], squeezed[idx_variables, idx_nodes] =
        compute_benchmarks_2d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "2D_nVarTotal_nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential sequential_prealloc squeezed

  # n_variables_interp = n_variables_total, n_nodes_out = 2*n_nodes_in
  # visualization
  title = "n_variables_interp = 2*n_variables_total, n_nodes_out = n_nodes_in"
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = n_variables_total
      n_nodes_out = 2 * n_nodes_in
      sequential[idx_variables, idx_nodes], sequential_prealloc[idx_variables, idx_nodes], squeezed[idx_variables, idx_nodes] =
        compute_benchmarks_2d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "2D_nVarTotal_2nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential sequential_prealloc squeezed

  # n_variables_interp = 1, n_nodes_out = n_nodes_in
  # blending
  title = "n_variables_interp = 1, n_nodes_out = n_nodes_in"
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = 1
      n_nodes_out = n_nodes_in
      sequential[idx_variables, idx_nodes], sequential_prealloc[idx_variables, idx_nodes], squeezed[idx_variables, idx_nodes] =
        compute_benchmarks_2d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "2D_nVar1_nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential sequential_prealloc squeezed

  return nothing
end

# TODO
compute_benchmarks_2d(1:10, 2:10)


###################################################################################################
# 3D versions

function multiply_coordinatewise_sequential!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  tmp1 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in, n_nodes_in)
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp1[v, i, j, k] += vandermonde[i, ii] * data_in[v, ii, j, k]
      end
    end
  end

  # Interpolate in y-direction
  tmp2 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_out, n_nodes_in)
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_out, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_variables
        tmp2[v, i, j, k] += vandermonde[j, jj] * tmp1[v, i, jj, k]
      end
    end
  end

  # Interpolate in z-direction
  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for kk in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j, k] += vandermonde[k, kk] * tmp2[v, i, j, kk]
      end
    end
  end

  return data_out
end

@muladd function multiply_coordinatewise_sequential_muladd!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  tmp1 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in, n_nodes_in)
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp1[v, i, j, k] += vandermonde[i, ii] * data_in[v, ii, j, k]
      end
    end
  end

  # Interpolate in y-direction
  tmp2 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_out, n_nodes_in)
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_out, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_variables
        tmp2[v, i, j, k] += vandermonde[j, jj] * tmp1[v, i, jj, k]
      end
    end
  end

  # Interpolate in z-direction
  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for kk in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j, k] += vandermonde[k, kk] * tmp2[v, i, j, kk]
      end
    end
  end

  return data_out
end

function multiply_coordinatewise_sequential_simd!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  tmp1 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in, n_nodes_in)
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_in, i in 1:n_nodes_out
    @simd for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp1[v, i, j, k] += vandermonde[i, ii] * data_in[v, ii, j, k]
      end
    end
  end

  # Interpolate in y-direction
  tmp2 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_out, n_nodes_in)
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_out, i in 1:n_nodes_out
    @simd for jj in 1:n_nodes_in
      for v in 1:n_variables
        tmp2[v, i, j, k] += vandermonde[j, jj] * tmp1[v, i, jj, k]
      end
    end
  end

  # Interpolate in z-direction
  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    @simd for kk in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j, k] += vandermonde[k, kk] * tmp2[v, i, j, kk]
      end
    end
  end

  return data_out
end

@muladd function multiply_coordinatewise_sequential_simd_muladd!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  tmp1 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in, n_nodes_in)
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_in, i in 1:n_nodes_out
    @simd for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp1[v, i, j, k] += vandermonde[i, ii] * data_in[v, ii, j, k]
      end
    end
  end

  # Interpolate in y-direction
  tmp2 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_out, n_nodes_in)
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_out, i in 1:n_nodes_out
    @simd for jj in 1:n_nodes_in
      for v in 1:n_variables
        tmp2[v, i, j, k] += vandermonde[j, jj] * tmp1[v, i, jj, k]
      end
    end
  end

  # Interpolate in z-direction
  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    @simd for kk in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j, k] += vandermonde[k, kk] * tmp2[v, i, j, kk]
      end
    end
  end

  return data_out
end

function multiply_coordinatewise_sequential_avx!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables;
    tmp1=zeros(eltype(data_out), n_variables, size(vandermonde, 1), size(vandermonde, 2), size(vandermonde, 2)),
    tmp2=zeros(eltype(data_out), n_variables, size(vandermonde, 1), size(vandermonde, 1), size(vandermonde, 2))) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  # tmp1 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_in, n_nodes_in)
  @avx for k in 1:n_nodes_in, j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_variables
        tmp1[v, i, j, k] += vandermonde[i, ii] * data_in[v, ii, j, k]
      end
    end
  end

  # Interpolate in y-direction
  # tmp2 = zeros(eltype(data_out), n_variables, n_nodes_out, n_nodes_out, n_nodes_in)
  @avx for k in 1:n_nodes_in, j in 1:n_nodes_out, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_variables
        tmp2[v, i, j, k] += vandermonde[j, jj] * tmp1[v, i, jj, k]
      end
    end
  end

  # Interpolate in z-direction
  @avx for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for kk in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j, k] += vandermonde[k, kk] * tmp2[v, i, j, kk]
      end
    end
  end

  return data_out
end


function multiply_coordinatewise_squeezed!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_variables
      acc = zero(eltype(data_out))
      for kk in 1:n_nodes_in, jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * vandermonde[k, kk] * data_in[v, ii, jj, kk]
      end
      data_out[v, i, j, k] = acc
    end
  end

  return data_out
end

@muladd function multiply_coordinatewise_squeezed_muladd!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_variables
      acc = zero(eltype(data_out))
      for kk in 1:n_nodes_in, jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * vandermonde[k, kk] * data_in[v, ii, jj, kk]
      end
      data_out[v, i, j, k] = acc
    end
  end

  return data_out
end

function multiply_coordinatewise_squeezed_simd!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    @simd for v in 1:n_variables
      acc = zero(eltype(data_out))
      for kk in 1:n_nodes_in, jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * vandermonde[k, kk] * data_in[v, ii, jj, kk]
      end
      data_out[v, i, j, k] = acc
    end
  end

  return data_out
end

@muladd function multiply_coordinatewise_squeezed_simd_muladd!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    @simd for v in 1:n_variables
      acc = zero(eltype(data_out))
      for kk in 1:n_nodes_in, jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * vandermonde[k, kk] * data_in[v, ii, jj, kk]
      end
      data_out[v, i, j, k] = acc
    end
  end

  return data_out
end

function multiply_coordinatewise_squeezed_avx!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in, 1)  >= n_variables) &&
               (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @avx for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_variables
      acc = zero(eltype(data_out))
      for kk in 1:n_nodes_in, jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * vandermonde[k, kk] * data_in[v, ii, jj, kk]
      end
      data_out[v, i, j, k] = acc
    end
  end

  return data_out
end


function run_benchmarks_3d(;n_variables_total=4, n_variables_interp=n_variables_total, n_nodes_in=4, n_nodes_out=2*n_nodes_in)
  data_in  = randn(n_variables_total, n_nodes_in,  n_nodes_in,  n_nodes_in)
  data_out = randn(n_variables_total, n_nodes_out, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde_static  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)

  println("multiply_coordinatewise_sequential!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_sequential!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("vandermonde_static")
  display(@benchmark multiply_coordinatewise_sequential!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_sequential_muladd!")
  display(@benchmark multiply_coordinatewise_sequential_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_sequential_simd!")
  display(@benchmark multiply_coordinatewise_sequential_simd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_sequential_simd_muladd!")
  display(@benchmark multiply_coordinatewise_sequential_simd_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_sequential_avx!")
  display(@benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed!")
  display(@benchmark multiply_coordinatewise_squeezed!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed_muladd!")
  display(@benchmark multiply_coordinatewise_squeezed_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed_simd!")
  display(@benchmark multiply_coordinatewise_squeezed_simd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed_simd_muladd!")
  display(@benchmark multiply_coordinatewise_squeezed_simd_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("multiply_coordinatewise_squeezed_avx!")
  display(@benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  nothing
end

# TODO
run_benchmarks_3d(n_variables_total=5, n_variables_interp=5, n_nodes_in=4, n_nodes_out=4)


function compute_benchmarks_3d(;n_variables_total=4, n_variables_interp=n_variables_total, n_nodes_in=4, n_nodes_out=2*n_nodes_in)
  data_in  = randn(n_variables_total, n_nodes_in, n_nodes_in,  n_nodes_in)
  data_out = randn(n_variables_interp, n_nodes_out, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)
  tmp1 = zeros(eltype(data_out), n_variables_interp, n_nodes_out, n_nodes_in, n_nodes_in)
  tmp2 = zeros(eltype(data_out), n_variables_interp, n_nodes_out, n_nodes_out, n_nodes_in)

  println("n_variables_total = ", n_variables_total, "; n_variables_interp = ", n_variables_interp,
          "; n_nodes_in = ", n_nodes_in, "; n_nodes_out = ", n_nodes_out)
  sequential = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde), $(n_variables_interp))
  sequential_prealloc = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde), $(n_variables_interp), tmp1=$(tmp1), tmp2=$(tmp2))
  # TODO: The @avvx version is significantly slower than the @simd version - why?
  #       maybe related to https://github.com/chriselrod/LoopVectorization.jl/issues/126
  # squeezed = @benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde), $(n_variables_interp))
  squeezed = @benchmark multiply_coordinatewise_squeezed_simd!($(data_out), $(data_in), $(vandermonde), $(n_variables_interp))

  return time(median(sequential)), time(median(sequential_prealloc)), time(median(squeezed))
end

function compute_benchmarks_3d(n_variables_total_list, n_nodes_in_list)
  sequential          = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_prealloc = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  squeezed            = zeros(length(n_variables_total_list), length(n_nodes_in_list))

  # n_variables_interp = n_variables_total, n_nodes_out = n_nodes_in
  # mortar
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = n_variables_total
      n_nodes_out = n_nodes_in
      sequential[idx_variables, idx_nodes], sequential_prealloc[idx_variables, idx_nodes], squeezed[idx_variables, idx_nodes] =
        compute_benchmarks_3d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "3D_nVarTotal_nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential sequential_prealloc squeezed

  # n_variables_interp = n_variables_total, n_nodes_out = 2*n_nodes_in
  # visualization
  title = "n_variables_interp = 2*n_variables_total, n_nodes_out = n_nodes_in"
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = n_variables_total
      n_nodes_out = 2 * n_nodes_in
      sequential[idx_variables, idx_nodes], sequential_prealloc[idx_variables, idx_nodes], squeezed[idx_variables, idx_nodes] =
        compute_benchmarks_3d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "3D_nVarTotal_2nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential sequential_prealloc squeezed

  # n_variables_interp = 1, n_nodes_out = n_nodes_in
  # blending
  title = "n_variables_interp = 1, n_nodes_out = n_nodes_in"
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = 1
      n_nodes_out = n_nodes_in
      sequential[idx_variables, idx_nodes], sequential_prealloc[idx_variables, idx_nodes], squeezed[idx_variables, idx_nodes] =
        compute_benchmarks_3d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "3D_nVar1_nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential sequential_prealloc squeezed

  return nothing
end


# TODO
compute_benchmarks_3d(1:10, 2:10)

