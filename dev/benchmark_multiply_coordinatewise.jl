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
  data_out .= zero(eltype(data_out))

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
  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
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
  data_out .= zero(eltype(data_out))

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
  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
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
  data_out .= zero(eltype(data_out))

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
  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
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
  data_out .= zero(eltype(data_out))

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
  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
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
  data_out .= zero(eltype(data_out))

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
  @avx for j in 1:n_nodes_out, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_variables
        data_out[v, i, j] += vandermonde[j, jj] * tmp[v, i, jj]
      end
    end
  end

  return data_out
end

@generated function multiply_coordinatewise_sequential_nexpr!(
    data_out::AbstractArray{T, 3}, data_in::AbstractArray{T, 3}, vandermonde::SMatrix{n_nodes_out,n_nodes_in}, ::Val{n_variables}) where {T,n_nodes_out,n_nodes_in,n_variables}
  quote
    @boundscheck begin
      inbounds = (size(data_out, 1) >= $n_variables) &&
                 (size(data_out, 2) == size(data_out, 3) == $n_nodes_out) &&
                 (size(data_in, 1) >= $n_variables) &&
                 (size(data_in, 2) == size(data_in, 3) == $n_nodes_in)
      inbounds || throw(BoundsError())
    end

    # Interpolate in x-direction
    # tmp1 = zeros(eltype(data_out), n_variables, $n_nodes_out, $n_nodes_in)
    @inbounds Base.Cartesian.@nexprs $n_nodes_in j -> begin
      Base.Cartesian.@nexprs $n_nodes_out i -> begin
        Base.Cartesian.@nexprs $n_variables v -> begin
          tmp1_v_i_j = zero(eltype(data_out))
          Base.Cartesian.@nexprs $n_nodes_in ii -> begin
            tmp1_v_i_j += vandermonde[i, ii] * data_in[v, ii, j]
          end
        end
      end
    end

    # Interpolate in y-direction
    @inbounds Base.Cartesian.@nexprs $n_nodes_out j -> begin
      Base.Cartesian.@nexprs $n_nodes_out i -> begin
        Base.Cartesian.@nexprs $n_variables v -> begin
          tmp2_v_i_j = zero(eltype(data_out))
          Base.Cartesian.@nexprs $n_nodes_in jj -> begin
            tmp2_v_i_j += vandermonde[j, jj] * tmp1_v_i_jj
          end
          data_out[v, i, j] = tmp2_v_i_j
        end
      end
    end

    return data_out
  end
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
  data_out = randn(n_variables_interp, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde_static  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)

  println("\n\n# 2D   ", "#"^70)

  println("\nmultiply_coordinatewise_sequential!")
  println("vandermonde_dynamic")
  multiply_coordinatewise_sequential!(data_out, data_in, vandermonde_dynamic, n_variables_interp)
  data_out_copy = copy(data_out)
  display(@benchmark multiply_coordinatewise_sequential!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  println("\nvandermonde_static")
  multiply_coordinatewise_sequential!(data_out, data_in, vandermonde_static, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_sequential!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  # println("\nmultiply_coordinatewise_sequential_muladd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_sequential_muladd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_sequential_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  # println("\nmultiply_coordinatewise_sequential_simd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_sequential_simd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_sequential_simd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  # println("\nmultiply_coordinatewise_sequential_simd_muladd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_sequential_simd_muladd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_sequential_simd_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  println("\nmultiply_coordinatewise_sequential_avx!")
  println("vandermonde_dynamic")
  multiply_coordinatewise_sequential_avx!(data_out, data_in, vandermonde_dynamic, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  println("\nvandermonde_static")
  multiply_coordinatewise_sequential_avx!(data_out, data_in, vandermonde_static, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("\nmultiply_coordinatewise_sequential_nexpr!")
  println("\nvandermonde_static")
  multiply_coordinatewise_sequential_nexpr!(data_out, data_in, vandermonde_static, Val(n_variables_interp))
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_sequential_nexpr!($(data_out), $(data_in), $(vandermonde_static), $(Val(n_variables_interp))))
  println()


  println("\nmultiply_coordinatewise_squeezed!")
  println("vandermonde_dynamic")
  multiply_coordinatewise_squeezed!(data_out, data_in, vandermonde_dynamic, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_squeezed!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  println("\nvandermonde_static")
  multiply_coordinatewise_squeezed!(data_out, data_in, vandermonde_static, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_squeezed!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  # println("\nmultiply_coordinatewise_squeezed_muladd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_squeezed_muladd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_squeezed_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  # println("\nmultiply_coordinatewise_squeezed_simd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_squeezed_simd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_squeezed_simd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  # println("\nmultiply_coordinatewise_squeezed_simd_muladd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_squeezed_simd_muladd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_squeezed_simd_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  println("\nmultiply_coordinatewise_squeezed_avx!")
  println("vandermonde_dynamic")
  multiply_coordinatewise_squeezed_avx!(data_out, data_in, vandermonde_dynamic, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  println("\nvandermonde_static")
  multiply_coordinatewise_squeezed_avx!(data_out, data_in, vandermonde_static, n_variables_interp)
  @assert data_out ≈ data_out_copy
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
  vandermonde_static = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)
  tmp = zeros(eltype(data_out), n_variables_interp, n_nodes_out, n_nodes_in)

  println("n_variables_total = ", n_variables_total, "; n_variables_interp = ", n_variables_interp,
          "; n_nodes_in = ", n_nodes_in, "; n_nodes_out = ", n_nodes_out)
  sequential_dynamic = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp))
  sequential_static  = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp))
  #FIXME sequential_nexpr   = @benchmark multiply_coordinatewise_sequential_nexpr!($(data_out), $(data_in), $(vandermonde_static), $(Val(n_variables_interp)))
  sequential_dynamic_prealloc = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp), tmp=$(tmp))
  sequential_static_prealloc  = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp), tmp=$(tmp))
  squeezed_dynamic = @benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp))
  squeezed_static  = @benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp))

  return time(median(sequential_dynamic)),
         time(median(sequential_static)),
         NaN, #FIXME time(median(sequential_nexpr)),
         time(median(sequential_dynamic_prealloc)),
         time(median(sequential_static_prealloc)),
         time(median(squeezed_dynamic)),
         time(median(squeezed_static))
end

function compute_benchmarks_2d(n_variables_total_list, n_nodes_in_list)
  sequential_dynamic          = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_static           = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_nexpr            = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_dynamic_prealloc = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_static_prealloc  = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  squeezed_dynamic            = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  squeezed_static             = zeros(length(n_variables_total_list), length(n_nodes_in_list))

  # n_variables_interp = n_variables_total, n_nodes_out = n_nodes_in
  # mortar
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = n_variables_total
      n_nodes_out = n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_2d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "2D_nVarTotal_nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  # n_variables_interp = n_variables_total, n_nodes_out = 2*n_nodes_in
  # visualization
  title = "n_variables_interp = 2*n_variables_total, n_nodes_out = n_nodes_in"
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = n_variables_total
      n_nodes_out = 2 * n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_2d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "2D_nVarTotal_2nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  # n_variables_interp = 1, n_nodes_out = n_nodes_in
  # blending
  title = "n_variables_interp = 1, n_nodes_out = n_nodes_in"
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = 1
      n_nodes_out = n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_2d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "2D_nVar1_nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

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
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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

@generated function multiply_coordinatewise_sequential_nexpr!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde::SMatrix{n_nodes_out,n_nodes_in}, ::Val{n_variables}) where {T,n_nodes_out,n_nodes_in,n_variables}
  quote
    @boundscheck begin
      inbounds = (size(data_out, 1) >= $n_variables) &&
                (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == $n_nodes_out) &&
                (size(data_in, 1) >= $n_variables) &&
                (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == $n_nodes_in)
      inbounds || throw(BoundsError())
    end

    # Interpolate in x-direction
    # tmp1 = zeros(eltype(data_out), n_variables, $n_nodes_out, $n_nodes_in, $n_nodes_in)
    @inbounds Base.Cartesian.@nexprs $n_nodes_in k -> begin
      Base.Cartesian.@nexprs $n_nodes_in j -> begin
        Base.Cartesian.@nexprs $n_nodes_out i -> begin
          Base.Cartesian.@nexprs $n_variables v -> begin
            tmp1_v_i_j_k = zero(eltype(data_out))
            Base.Cartesian.@nexprs $n_nodes_in ii -> begin
              tmp1_v_i_j_k += vandermonde[i, ii] * data_in[v, ii, j, k]
            end
          end
        end
      end
    end

    # Interpolate in y-direction
    # tmp2 = zeros(eltype(data_out), n_variables, $n_nodes_out, $n_nodes_out, $n_nodes_in)
    @inbounds Base.Cartesian.@nexprs $n_nodes_in k -> begin
      Base.Cartesian.@nexprs $n_nodes_out j -> begin
        Base.Cartesian.@nexprs $n_nodes_out i -> begin
          Base.Cartesian.@nexprs $n_variables v -> begin
            tmp2_v_i_j_k = zero(eltype(data_out))
            Base.Cartesian.@nexprs $n_nodes_in jj -> begin
              tmp2_v_i_j_k += vandermonde[j, jj] * tmp1_v_i_jj_k
            end
          end
        end
      end
    end

    # Interpolate in z-direction
    @inbounds Base.Cartesian.@nexprs $n_nodes_out k -> begin
      Base.Cartesian.@nexprs $n_nodes_out j -> begin
        Base.Cartesian.@nexprs $n_nodes_out i -> begin
          Base.Cartesian.@nexprs $n_variables v -> begin
            tmp3_v_i_j_k = zero(eltype(data_out))
            Base.Cartesian.@nexprs $n_nodes_in kk -> begin
              tmp3_v_i_j_k += vandermonde[k, kk] * tmp2_v_i_j_kk
            end
            data_out[v, i, j, k] = tmp3_v_i_j_k
          end
        end
      end
    end

    return data_out
  end
end


function multiply_coordinatewise_squeezed!(
    data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)

  @boundscheck begin
    inbounds = (size(data_out, 1) >= n_variables) &&
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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
               (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
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

# requires using Tullio
# function multiply_coordinatewise_squeezed_tullio!(
#     data_out::AbstractArray{T, 4}, data_in::AbstractArray{T, 4}, vandermonde, n_variables) where T
#   n_nodes_out = size(vandermonde, 1)
#   n_nodes_in  = size(vandermonde, 2)

#   @boundscheck begin
#     inbounds = (size(data_out, 1) >= n_variables) &&
#                (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
#                (size(data_in, 1)  >= n_variables) &&
#                (size(data_in, 2) == size(data_in, 3) == size(data_in, 4) == n_nodes_in)
#     inbounds || throw(BoundsError())
#   end

#   @tullio data_out[v, i, j, k] = vandermonde[i, ii] * vandermonde[j, jj] * vandermonde[k, kk] * data_in[v, ii, jj, kk]

#   return data_out
# end


function run_benchmarks_3d(;n_variables_total=4, n_variables_interp=n_variables_total, n_nodes_in=4, n_nodes_out=2*n_nodes_in)
  data_in  = randn(n_variables_total, n_nodes_in,  n_nodes_in,  n_nodes_in)
  data_out = randn(n_variables_interp, n_nodes_out, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde_static  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)

  println("\n\n# 3D   ", "#"^70)

  println("\nmultiply_coordinatewise_sequential!")
  println("vandermonde_dynamic")
  multiply_coordinatewise_sequential!(data_out, data_in, vandermonde_dynamic, n_variables_interp)
  data_out_copy = copy(data_out)
  display(@benchmark multiply_coordinatewise_sequential!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  println("\nvandermonde_static")
  multiply_coordinatewise_sequential!(data_out, data_in, vandermonde_static, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_sequential!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  # println("\nmultiply_coordinatewise_sequential_muladd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_sequential_muladd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_sequential_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  # println("\nmultiply_coordinatewise_sequential_simd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_sequential_simd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_sequential_simd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  # println("\nmultiply_coordinatewise_sequential_simd_muladd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_sequential_simd_muladd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_sequential_simd_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  println("\nmultiply_coordinatewise_sequential_avx!")
  println("vandermonde_dynamic")
  multiply_coordinatewise_sequential_avx!(data_out, data_in, vandermonde_dynamic, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  println("\nvandermonde_static")
  multiply_coordinatewise_sequential_avx!(data_out, data_in, vandermonde_static, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  println("\nmultiply_coordinatewise_sequential_nexpr!")
  println("\nvandermonde_static")
  multiply_coordinatewise_sequential_nexpr!(data_out, data_in, vandermonde_static, Val(n_variables_interp))
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_sequential_nexpr!($(data_out), $(data_in), $(vandermonde_static), $(Val(n_variables_interp))))
  println()

  println("\nmultiply_coordinatewise_squeezed!")
  println("vandermonde_dynamic")
  multiply_coordinatewise_squeezed!(data_out, data_in, vandermonde_dynamic, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_squeezed!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  println("\nvandermonde_static")
  multiply_coordinatewise_squeezed!(data_out, data_in, vandermonde_static, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_squeezed!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  println()

  # println("\nmultiply_coordinatewise_squeezed_muladd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_squeezed_muladd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_squeezed_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  # println("\nmultiply_coordinatewise_squeezed_simd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_squeezed_simd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_squeezed_simd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  # println("\nmultiply_coordinatewise_squeezed_simd_muladd!")
  # println("vandermonde_dynamic")
  # display(@benchmark multiply_coordinatewise_squeezed_simd_muladd!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  # println("\nvandermonde_static")
  # display(@benchmark multiply_coordinatewise_squeezed_simd_muladd!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp)))
  # println()

  println("\nmultiply_coordinatewise_squeezed_avx!")
  println("vandermonde_dynamic")
  multiply_coordinatewise_squeezed_avx!(data_out, data_in, vandermonde_dynamic, n_variables_interp)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp)))
  println("\nvandermonde_static")
  multiply_coordinatewise_squeezed_avx!(data_out, data_in, vandermonde_static, n_variables_interp)
  @assert data_out ≈ data_out_copy
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
  vandermonde_static  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)
  tmp1 = zeros(eltype(data_out), n_variables_interp, n_nodes_out, n_nodes_in, n_nodes_in)
  tmp2 = zeros(eltype(data_out), n_variables_interp, n_nodes_out, n_nodes_out, n_nodes_in)

  println("n_variables_total = ", n_variables_total, "; n_variables_interp = ", n_variables_interp,
          "; n_nodes_in = ", n_nodes_in, "; n_nodes_out = ", n_nodes_out)
  sequential_dynamic = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp))
  sequential_static  = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp))
  #FIXME sequential_nexpr   = @benchmark multiply_coordinatewise_sequential_nexpr!($(data_out), $(data_in), $(vandermonde_static), $(Val(n_variables_interp)))
  sequential_dynamic_prealloc = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp), tmp1=$(tmp1), tmp2=$(tmp2))
  sequential_static_prealloc  = @benchmark multiply_coordinatewise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp), tmp1=$(tmp1), tmp2=$(tmp2))
  squeezed_dynamic = @benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_dynamic), $(n_variables_interp))
  squeezed_static  = @benchmark multiply_coordinatewise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_static), $(n_variables_interp))

  return time(median(sequential_dynamic)),
         time(median(sequential_static)),
         NaN, #FIXME time(median(sequential_nexpr)),
         time(median(sequential_dynamic_prealloc)),
         time(median(sequential_static_prealloc)),
         time(median(squeezed_dynamic)),
         time(median(squeezed_static))
end

function compute_benchmarks_3d(n_variables_total_list, n_nodes_in_list)
  sequential_dynamic          = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_static           = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_nexpr            = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_dynamic_prealloc = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  sequential_static_prealloc  = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  squeezed_dynamic            = zeros(length(n_variables_total_list), length(n_nodes_in_list))
  squeezed_static             = zeros(length(n_variables_total_list), length(n_nodes_in_list))

  # n_variables_interp = n_variables_total, n_nodes_out = n_nodes_in
  # mortar
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = n_variables_total
      n_nodes_out = n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_3d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "3D_nVarTotal_nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  # n_variables_interp = n_variables_total, n_nodes_out = 2*n_nodes_in
  # visualization
  title = "n_variables_interp = 2*n_variables_total, n_nodes_out = n_nodes_in"
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = n_variables_total
      n_nodes_out = 2 * n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_3d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "3D_nVarTotal_2nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  # n_variables_interp = 1, n_nodes_out = n_nodes_in
  # blending
  title = "n_variables_interp = 1, n_nodes_out = n_nodes_in"
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_variables_total) in enumerate(n_variables_total_list)
      n_variables_interp = 1
      n_nodes_out = n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_3d(;n_variables_total=n_variables_total, n_variables_interp=n_variables_interp, n_nodes_in=n_nodes_in, n_nodes_out=n_nodes_out)
    end
  end
  BSON.@save "3D_nVar1_nNodesIn.bson" n_variables_total_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  return nothing
end


# TODO
compute_benchmarks_3d(1:10, 2:10)
