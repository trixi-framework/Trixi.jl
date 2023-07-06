# Disable formatting this file since it contains highly unusual formatting for better
# readability
#! format: off

import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using BenchmarkTools
using BSON
using LoopVectorization
using MuladdMacro
using StaticArrays
using Tullio

###################################################################################################
# 2D versions
function multiply_dimensionwise_sequential!(
    data_out::AbstractArray{<:Any, 3}, data_in::AbstractArray{<:Any, 3}, vandermonde,
    tmp1=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 2)))
  n_vars      = size(data_out, 1)
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in,  1) == n_vars) &&
               (size(data_in,  2) == size(data_in,  3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  @inbounds for j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_vars
        tmp1[v, i, j] += vandermonde[i, ii] * data_in[v, ii, j]
      end
    end
  end

  # Interpolate in y-direction
  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_vars
        data_out[v, i, j] += vandermonde[j, jj] * tmp1[v, i, jj]
      end
    end
  end

  return data_out
end

function multiply_dimensionwise_sequential_avx!(
    data_out::AbstractArray{<:Any, 3}, data_in::AbstractArray{<:Any, 3}, vandermonde,
    tmp1=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 2)))
  n_vars      = size(data_out, 1)
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in,  1) == n_vars) &&
               (size(data_in,  2) == size(data_in,  3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  @avx for j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_vars
        tmp1[v, i, j] += vandermonde[i, ii] * data_in[v, ii, j]
      end
    end
  end

  # Interpolate in y-direction
  @avx for j in 1:n_nodes_out, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_vars
        data_out[v, i, j] += vandermonde[j, jj] * tmp1[v, i, jj]
      end
    end
  end

  return data_out
end

function multiply_dimensionwise_sequential_tullio!(
    data_out::AbstractArray{<:Any, 3}, data_in::AbstractArray{<:Any, 3}, vandermonde,
    tmp1=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 2)))

  # Interpolate in x-direction
  @tullio threads=false tmp1[v, i, j] = vandermonde[i, ii] * data_in[v, ii, j]

  # Interpolate in y-direction
  @tullio threads=false data_out[v, i, j] = vandermonde[j, jj] * tmp1[v, i, jj]

  return data_out
end

@generated function multiply_dimensionwise_sequential_nexpr!(
    data_out::AbstractArray{<:Any, 3}, data_in::AbstractArray{<:Any, 3}, vandermonde::SMatrix{n_nodes_out,n_nodes_in}, ::Val{n_vars}) where {n_nodes_out, n_nodes_in, n_vars}
  quote
    @boundscheck begin
      inbounds = (size(data_out, 1) == $n_vars) &&
                 (size(data_out, 2) == size(data_out, 3) == $n_nodes_out) &&
                 (size(data_in,  1) == $n_vars) &&
                 (size(data_in,  2) == size(data_in,  3) == $n_nodes_in)
      inbounds || throw(BoundsError())
    end

    # Interpolate in x-direction
    @inbounds @muladd Base.Cartesian.@nexprs $n_nodes_in j -> begin
      Base.Cartesian.@nexprs $n_nodes_out i -> begin
        Base.Cartesian.@nexprs $n_vars v -> begin
          tmp1_v_i_j = zero(eltype(data_out))
          Base.Cartesian.@nexprs $n_nodes_in ii -> begin
            tmp1_v_i_j += vandermonde[i, ii] * data_in[v, ii, j]
          end
        end
      end
    end

    # Interpolate in y-direction
    @inbounds @muladd Base.Cartesian.@nexprs $n_nodes_out j -> begin
      Base.Cartesian.@nexprs $n_nodes_out i -> begin
        Base.Cartesian.@nexprs $n_vars v -> begin
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


function multiply_dimensionwise_squeezed!(
    data_out::AbstractArray{<:Any, 3}, data_in::AbstractArray{<:Any, 3}, vandermonde)
  n_vars      = size(data_out, 1)
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in,  1) == n_vars) &&
               (size(data_in,  2) == size(data_in,  3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_vars
      acc = zero(eltype(data_out))
      for jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii]*  vandermonde[j, jj] * data_in[v, ii, jj]
      end
      data_out[v, i, j] = acc
    end
  end

  return data_out
end

function multiply_dimensionwise_squeezed_avx!(
    data_out::AbstractArray{<:Any, 3}, data_in::AbstractArray{<:Any, 3}, vandermonde)
  n_vars      = size(data_out, 1)
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 2) == size(data_out, 3) == n_nodes_out) &&
               (size(data_in,  1) == n_vars) &&
               (size(data_in,  2) == size(data_in,  3) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @avx for j in 1:n_nodes_out, i in 1:n_nodes_out
    for v in 1:n_vars
      acc = zero(eltype(data_out))
      for jj in 1:n_nodes_in, ii in 1:n_nodes_in
        acc += vandermonde[i, ii] * vandermonde[j, jj] * data_in[v, ii, jj]
      end
      data_out[v, i, j] = acc
    end
  end

  return data_out
end

function multiply_dimensionwise_squeezed_tullio!(
    data_out::AbstractArray{<:Any, 3}, data_in::AbstractArray{<:Any, 3}, vandermonde)

  @tullio threads=false data_out[v, i, j] = vandermonde[i, ii] * vandermonde[j, jj] * data_in[v, ii, jj]

  return data_out
end


function run_benchmarks_2d(n_vars=4, n_nodes_in=4, n_nodes_out=2*n_nodes_in)
  data_in  = randn(n_vars, n_nodes_in,  n_nodes_in)
  data_out = randn(n_vars, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde_static  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)
  vandermonde_mmatrix = MMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)

  println("\n\n# 2D   ", "#"^70)
  println("n_vars      = ", n_vars)
  println("n_nodes_in  = ", n_nodes_in)
  println("n_nodes_out = ", n_nodes_out)
  println()

  println("\n","multiply_dimensionwise_sequential!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_sequential!(data_out, data_in, vandermonde_dynamic)
  data_out_copy = copy(data_out)
  display(@benchmark multiply_dimensionwise_sequential!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_sequential!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential!($(data_out), $(data_in), $(vandermonde_static)))
  println("\n", "vandermonde_mmatrix")
  multiply_dimensionwise_sequential!(data_out, data_in, vandermonde_mmatrix)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential!($(data_out), $(data_in), $(vandermonde_mmatrix)))
  println()

  println("\n","multiply_dimensionwise_sequential_avx!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_sequential_avx!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_avx!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_sequential_avx!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static)))
  println("\n", "vandermonde_mmatrix")
  multiply_dimensionwise_sequential_avx!(data_out, data_in, vandermonde_mmatrix)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_avx!($(data_out), $(data_in), $(vandermonde_mmatrix)))
  println()

  println("\n","multiply_dimensionwise_sequential_tullio!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_sequential_tullio!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_sequential_tullio!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_static)))
  println("\n", "vandermonde_mmatrix")
  multiply_dimensionwise_sequential_tullio!(data_out, data_in, vandermonde_mmatrix)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_mmatrix)))
  println()

  println("\n","multiply_dimensionwise_sequential_nexpr!")
  println("vandermonde_static")
  multiply_dimensionwise_sequential_nexpr!(data_out, data_in, vandermonde_static, Val(n_vars))
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_nexpr!($(data_out), $(data_in), $(vandermonde_static), $(Val(n_vars))))
  println()


  println("\n","multiply_dimensionwise_squeezed!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_squeezed!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_squeezed!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed!($(data_out), $(data_in), $(vandermonde_static)))
  println("\n", "vandermonde_mmatrix")
  multiply_dimensionwise_squeezed!(data_out, data_in, vandermonde_mmatrix)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed!($(data_out), $(data_in), $(vandermonde_mmatrix)))
  println()

  println("\n","multiply_dimensionwise_squeezed_avx!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_squeezed_avx!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_squeezed_avx!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_static)))
  println("\n", "vandermonde_mmatrix")
  multiply_dimensionwise_squeezed_avx!(data_out, data_in, vandermonde_mmatrix)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_mmatrix)))
  println()

  println("\n","multiply_dimensionwise_squeezed_tullio!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_squeezed_tullio!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_squeezed_tullio!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_static)))
  println("\n", "vandermonde_mmatrix")
  multiply_dimensionwise_squeezed_tullio!(data_out, data_in, vandermonde_mmatrix)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_mmatrix)))
  println()

  nothing
end

# TODO
run_benchmarks_2d(4, 4, 4) # n_vars, n_nodes_in, n_nodes_out


function compute_benchmarks_2d(n_vars, n_nodes_in, n_nodes_out)
  data_in  = randn(n_vars, n_nodes_in,  n_nodes_in)
  data_out = randn(n_vars, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde_static = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)
  tmp1 = zeros(eltype(data_out), n_vars, n_nodes_out, n_nodes_in)

  println("n_vars = ", n_vars, "; n_nodes_in = ", n_nodes_in, "; n_nodes_out = ", n_nodes_out)
  sequential_dynamic = @benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_dynamic))
  sequential_static  = @benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_static))
  #FIXME sequential_nexpr   = @benchmark multiply_dimensionwise_sequential_nexpr!($(data_out), $(data_in), $(vandermonde_static), $(Val(n_vars)))
  sequential_dynamic_prealloc = @benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_dynamic), $(tmp1))
  sequential_static_prealloc  = @benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_static), $(tmp1))
  squeezed_dynamic = @benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_dynamic))
  squeezed_static  = @benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_static))

  return time(median(sequential_dynamic)),
         time(median(sequential_static)),
         NaN, #FIXME time(median(sequential_nexpr)),
         time(median(sequential_dynamic_prealloc)),
         time(median(sequential_static_prealloc)),
         time(median(squeezed_dynamic)),
         time(median(squeezed_static))
end

function compute_benchmarks_2d(n_vars_list, n_nodes_in_list)
  sequential_dynamic          = zeros(length(n_vars_list), length(n_nodes_in_list))
  sequential_static           = zeros(length(n_vars_list), length(n_nodes_in_list))
  sequential_nexpr            = zeros(length(n_vars_list), length(n_nodes_in_list))
  sequential_dynamic_prealloc = zeros(length(n_vars_list), length(n_nodes_in_list))
  sequential_static_prealloc  = zeros(length(n_vars_list), length(n_nodes_in_list))
  squeezed_dynamic            = zeros(length(n_vars_list), length(n_nodes_in_list))
  squeezed_static             = zeros(length(n_vars_list), length(n_nodes_in_list))

  # n_nodes_out = n_nodes_in
  # mortar
  # superset of n_vars = 1, n_nodes_out = n_nodes_in, used for blending
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_vars) in enumerate(n_vars_list)
      n_nodes_out = n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_2d(n_vars, n_nodes_in, n_nodes_out)
    end
  end
  BSON.@save "2D_nVarTotal_nNodesIn.bson" n_vars_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  # n_nodes_out = 2*n_nodes_in
  # visualization
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_vars) in enumerate(n_vars_list)
      n_nodes_out = 2 * n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_2d(n_vars, n_nodes_in, n_nodes_out)
    end
  end
  BSON.@save "2D_nVarTotal_2nNodesIn.bson" n_vars_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  return nothing
end

# TODO
compute_benchmarks_2d(1:10, 2:10)


###################################################################################################
# 3D versions

function multiply_dimensionwise_sequential!(
    data_out::AbstractArray{<:Any, 4}, data_in::AbstractArray{<:Any, 4}, vandermonde,
    tmp1=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 2), size(vandermonde, 2)),
    tmp2=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 1), size(vandermonde, 2)))
  n_vars      = size(data_out, 1)
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
               (size(data_in,  1) == n_vars) &&
               (size(data_in,  2) == size(data_in,  3) == size(data_in,  4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_vars
        tmp1[v, i, j, k] += vandermonde[i, ii] * data_in[v, ii, j, k]
      end
    end
  end

  # Interpolate in y-direction
  @inbounds for k in 1:n_nodes_in, j in 1:n_nodes_out, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_vars
        tmp2[v, i, j, k] += vandermonde[j, jj] * tmp1[v, i, jj, k]
      end
    end
  end

  # Interpolate in z-direction
  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for kk in 1:n_nodes_in
      for v in 1:n_vars
        data_out[v, i, j, k] += vandermonde[k, kk] * tmp2[v, i, j, kk]
      end
    end
  end

  return data_out
end

function multiply_dimensionwise_sequential_avx!(
    data_out::AbstractArray{<:Any, 4}, data_in::AbstractArray{<:Any, 4}, vandermonde,
    tmp1=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 2), size(vandermonde, 2)),
    tmp2=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 1), size(vandermonde, 2)))
  n_vars      = size(data_out, 1)
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
               (size(data_in,  1) == n_vars) &&
               (size(data_in,  2) == size(data_in,  3) == size(data_in,  4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  # Interpolate in x-direction
  @avx for k in 1:n_nodes_in, j in 1:n_nodes_in, i in 1:n_nodes_out
    for ii in 1:n_nodes_in
      for v in 1:n_vars
        tmp1[v, i, j, k] += vandermonde[i, ii] * data_in[v, ii, j, k]
      end
    end
  end

  # Interpolate in y-direction
  @avx for k in 1:n_nodes_in, j in 1:n_nodes_out, i in 1:n_nodes_out
    for jj in 1:n_nodes_in
      for v in 1:n_vars
        tmp2[v, i, j, k] += vandermonde[j, jj] * tmp1[v, i, jj, k]
      end
    end
  end

  # Interpolate in z-direction
  @avx for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
    for kk in 1:n_nodes_in
      for v in 1:n_vars
        data_out[v, i, j, k] += vandermonde[k, kk] * tmp2[v, i, j, kk]
      end
    end
  end

  return data_out
end

function multiply_dimensionwise_sequential_tullio!(
    data_out::AbstractArray{<:Any, 4}, data_in::AbstractArray{<:Any, 4}, vandermonde,
    tmp1=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 2), size(vandermonde, 2)),
    tmp2=zeros(eltype(data_out), size(data_out, 1), size(vandermonde, 1), size(vandermonde, 1), size(vandermonde, 2)))

  # Interpolate in x-direction
  @tullio threads=false tmp1[v, i, j, k] = vandermonde[i, ii] * data_in[v, ii, j, k]

  # Interpolate in y-direction
  @tullio threads=false tmp2[v, i, j, k] = vandermonde[j, jj] * tmp1[v, i, jj, k]

  # Interpolate in z-direction
  @tullio threads=false data_out[v, i, j, k] = vandermonde[k, kk] * tmp2[v, i, j, kk]

  return data_out
end

@generated function multiply_dimensionwise_sequential_nexpr!(
    data_out::AbstractArray{<:Any, 4}, data_in::AbstractArray{<:Any, 4}, vandermonde::SMatrix{n_nodes_out,n_nodes_in}, ::Val{n_vars}) where {n_nodes_out, n_nodes_in, n_vars}
  quote
    @boundscheck begin
      inbounds = (size(data_out, 1) == $n_vars) &&
                 (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == $n_nodes_out) &&
                 (size(data_in,  1) == $n_vars) &&
                 (size(data_in,  2) == size(data_in,  3) == size(data_in,  4) == $n_nodes_in)
      inbounds || throw(BoundsError())
    end

    # Interpolate in x-direction
    @inbounds @muladd Base.Cartesian.@nexprs $n_nodes_in k -> begin
      Base.Cartesian.@nexprs $n_nodes_in j -> begin
        Base.Cartesian.@nexprs $n_nodes_out i -> begin
          Base.Cartesian.@nexprs $n_vars v -> begin
            tmp1_v_i_j_k = zero(eltype(data_out))
            Base.Cartesian.@nexprs $n_nodes_in ii -> begin
              tmp1_v_i_j_k += vandermonde[i, ii] * data_in[v, ii, j, k]
            end
          end
        end
      end
    end

    # Interpolate in y-direction
    @inbounds @muladd Base.Cartesian.@nexprs $n_nodes_in k -> begin
      Base.Cartesian.@nexprs $n_nodes_out j -> begin
        Base.Cartesian.@nexprs $n_nodes_out i -> begin
          Base.Cartesian.@nexprs $n_vars v -> begin
            tmp2_v_i_j_k = zero(eltype(data_out))
            Base.Cartesian.@nexprs $n_nodes_in jj -> begin
              tmp2_v_i_j_k += vandermonde[j, jj] * tmp1_v_i_jj_k
            end
          end
        end
      end
    end

    # Interpolate in z-direction
    @inbounds @muladd Base.Cartesian.@nexprs $n_nodes_out k -> begin
      Base.Cartesian.@nexprs $n_nodes_out j -> begin
        Base.Cartesian.@nexprs $n_nodes_out i -> begin
          Base.Cartesian.@nexprs $n_vars v -> begin
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


function multiply_dimensionwise_squeezed!(
    data_out::AbstractArray{<:Any, 4}, data_in::AbstractArray{<:Any, 4}, vandermonde)
  n_vars      = size(data_out, 1)
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
               (size(data_in,  1) == n_vars) &&
               (size(data_in,  2) == size(data_in,  3) == size(data_in,  4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @inbounds for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
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

function multiply_dimensionwise_squeezed_avx!(
    data_out::AbstractArray{<:Any, 4}, data_in::AbstractArray{<:Any, 4}, vandermonde)
  n_vars      = size(data_out, 1)
  n_nodes_out = size(vandermonde, 1)
  n_nodes_in  = size(vandermonde, 2)
  data_out .= zero(eltype(data_out))

  @boundscheck begin
    inbounds = (size(data_out, 2) == size(data_out, 3) == size(data_out, 4) == n_nodes_out) &&
               (size(data_in,  1) == n_vars) &&
               (size(data_in,  2) == size(data_in,  3) == size(data_in,  4) == n_nodes_in)
    inbounds || throw(BoundsError())
  end

  @avx for k in 1:n_nodes_out, j in 1:n_nodes_out, i in 1:n_nodes_out
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

function multiply_dimensionwise_squeezed_tullio!(
    data_out::AbstractArray{<:Any, 4}, data_in::AbstractArray{<:Any, 4}, vandermonde)

  @tullio threads=false data_out[v, i, j, k] = vandermonde[i, ii] * vandermonde[j, jj] * vandermonde[k, kk] * data_in[v, ii, jj, kk]

  return data_out
end


function run_benchmarks_3d(n_vars=4, n_nodes_in=4, n_nodes_out=2*n_nodes_in)
  data_in  = randn(n_vars, n_nodes_in,  n_nodes_in,  n_nodes_in)
  data_out = randn(n_vars, n_nodes_out, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde_static  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)

  println("\n\n# 3D   ", "#"^70)
  println("n_vars      = ", n_vars)
  println("n_nodes_in  = ", n_nodes_in)
  println("n_nodes_out = ", n_nodes_out)
  println()

  println("\n","multiply_dimensionwise_sequential!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_sequential!(data_out, data_in, vandermonde_dynamic)
  data_out_copy = copy(data_out)
  display(@benchmark multiply_dimensionwise_sequential!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_sequential!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential!($(data_out), $(data_in), $(vandermonde_static)))
  println()

  println("\n","multiply_dimensionwise_sequential_avx!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_sequential_avx!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_avx!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_sequential_avx!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_avx!($(data_out), $(data_in), $(vandermonde_static)))
  println()

  println("\n","multiply_dimensionwise_sequential_tullio!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_sequential_tullio!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_sequential_tullio!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_static)))
  println()

  println("\n","multiply_dimensionwise_sequential_nexpr!")
  println("vandermonde_static")
  multiply_dimensionwise_sequential_nexpr!(data_out, data_in, vandermonde_static, Val(n_vars))
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_sequential_nexpr!($(data_out), $(data_in), $(vandermonde_static), $(Val(n_vars))))
  println()

  println("\n","multiply_dimensionwise_squeezed!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_squeezed!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_squeezed!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed!($(data_out), $(data_in), $(vandermonde_static)))
  println()

  println("\n","multiply_dimensionwise_squeezed_avx!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_squeezed_avx!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_squeezed_avx!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_avx!($(data_out), $(data_in), $(vandermonde_static)))
  println()

  println("\n","multiply_dimensionwise_squeezed_tullio!")
  println("vandermonde_dynamic")
  multiply_dimensionwise_squeezed_tullio!(data_out, data_in, vandermonde_dynamic)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_dynamic)))
  println("\n", "vandermonde_static")
  multiply_dimensionwise_squeezed_tullio!(data_out, data_in, vandermonde_static)
  @assert data_out ≈ data_out_copy
  display(@benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_static)))
  println()

  nothing
end

# TODO
run_benchmarks_3d(5, 4, 4) # n_vars, n_nodes_in, n_nodes_out


function compute_benchmarks_3d(n_vars, n_nodes_in, n_nodes_out)
  data_in  = randn(n_vars, n_nodes_in,  n_nodes_in,  n_nodes_in)
  data_out = randn(n_vars, n_nodes_out, n_nodes_out, n_nodes_out)
  vandermonde_dynamic = randn(n_nodes_out, n_nodes_in)
  vandermonde_static  = SMatrix{n_nodes_out, n_nodes_in}(vandermonde_dynamic)
  tmp1 = zeros(eltype(data_out), n_vars, n_nodes_out, n_nodes_in,  n_nodes_in)
  tmp2 = zeros(eltype(data_out), n_vars, n_nodes_out, n_nodes_out, n_nodes_in)

  println("n_vars = ", n_vars, "; n_nodes_in = ", n_nodes_in, "; n_nodes_out = ", n_nodes_out)
  sequential_dynamic = @benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_dynamic))
  sequential_static  = @benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_static))
  #FIXME sequential_nexpr   = @benchmark multiply_dimensionwise_sequential_nexpr!($(data_out), $(data_in), $(vandermonde_static), $(Val(n_vars)))
  sequential_dynamic_prealloc = @benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_dynamic), $(tmp1), $(tmp2))
  sequential_static_prealloc  = @benchmark multiply_dimensionwise_sequential_tullio!($(data_out), $(data_in), $(vandermonde_static),  $(tmp1), $(tmp2))
  squeezed_dynamic = @benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_dynamic))
  squeezed_static  = @benchmark multiply_dimensionwise_squeezed_tullio!($(data_out), $(data_in), $(vandermonde_static))

  return time(median(sequential_dynamic)),
         time(median(sequential_static)),
         NaN, #FIXME time(median(sequential_nexpr)),
         time(median(sequential_dynamic_prealloc)),
         time(median(sequential_static_prealloc)),
         time(median(squeezed_dynamic)),
         time(median(squeezed_static))
end

function compute_benchmarks_3d(n_vars_list, n_nodes_in_list)
  sequential_dynamic          = zeros(length(n_vars_list), length(n_nodes_in_list))
  sequential_static           = zeros(length(n_vars_list), length(n_nodes_in_list))
  sequential_nexpr            = zeros(length(n_vars_list), length(n_nodes_in_list))
  sequential_dynamic_prealloc = zeros(length(n_vars_list), length(n_nodes_in_list))
  sequential_static_prealloc  = zeros(length(n_vars_list), length(n_nodes_in_list))
  squeezed_dynamic            = zeros(length(n_vars_list), length(n_nodes_in_list))
  squeezed_static             = zeros(length(n_vars_list), length(n_nodes_in_list))

  # n_nodes_out = n_nodes_in
  # mortar
  # superset of n_vars = 1, n_nodes_out = n_nodes_in, used for blending
  for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
    for (idx_variables, n_vars) in enumerate(n_vars_list)
      n_nodes_out = n_nodes_in
      sequential_dynamic[idx_variables, idx_nodes],
      sequential_static[idx_variables, idx_nodes],
      sequential_nexpr[idx_variables, idx_nodes],
      sequential_dynamic_prealloc[idx_variables, idx_nodes],
      sequential_static_prealloc[idx_variables, idx_nodes],
      squeezed_dynamic[idx_variables, idx_nodes],
      squeezed_static[idx_variables, idx_nodes] =
        compute_benchmarks_3d(n_vars, n_nodes_in, n_nodes_out)
    end
  end
  BSON.@save "3D_nVarTotal_nNodesIn.bson" n_vars_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  # TODO deactivated to save some time
  # # n_nodes_out = 2*n_nodes_in
  # # visualization
  # title = "n_vars = 2*n_vars, n_nodes_out = n_nodes_in"
  # for (idx_nodes, n_nodes_in) in enumerate(n_nodes_in_list)
  #   for (idx_variables, n_vars) in enumerate(n_vars_list)
  #     n_nodes_out = 2 * n_nodes_in
  #     sequential_dynamic[idx_variables, idx_nodes],
  #     sequential_static[idx_variables, idx_nodes],
  #     sequential_nexpr[idx_variables, idx_nodes],
  #     sequential_dynamic_prealloc[idx_variables, idx_nodes],
  #     sequential_static_prealloc[idx_variables, idx_nodes],
  #     squeezed_dynamic[idx_variables, idx_nodes],
  #     squeezed_static[idx_variables, idx_nodes] =
  #       compute_benchmarks_3d(n_vars, n_nodes_in, n_nodes_out)
  #   end
  # end
  # BSON.@save "3D_nVarTotal_2nNodesIn.bson" n_vars_list n_nodes_in_list sequential_dynamic sequential_static sequential_nexpr sequential_dynamic_prealloc sequential_static_prealloc squeezed_dynamic squeezed_static

  return nothing
end


# TODO
compute_benchmarks_3d(1:10, 2:10)
