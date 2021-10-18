# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin



# TODO: Upstream, LoopVectorization
#       At the time of writing, LoopVectorization.jl cannot handle this kind of
#       loop optimally when passing our custom functions `cons2entropy` and
#       `entropy2cons`. Thus, we need to insert the physics directly here to
#       get a significant runtime performance improvement.
function entropy_projection!(cache, u::StructArray{<:SVector{4, <:Union{Float32, Float64}}},
                             mesh::VertexMappedMesh, equations::CompressibleEulerEquations2D, dg::DGMultiFluxDiff{<:Polynomial})
  rd = dg.basis
  @unpack Vq = rd
  @unpack VhP, u_values, entropy_var_values = cache
  @unpack projected_entropy_var_values, entropy_projected_u_values = cache

  apply_to_each_field(mul_by!(Vq), u_values, u)

  # The following is semantically equivalent to
  # @threaded for i in Base.OneTo(length(u_values))
  #   entropy_var_values[i] = cons2entropy(u_values[i], equations)
  # end
  # but much more efficient due to explicit optimization via `@turbo` from
  # LoopVectorization.jl.
  @unpack gamma, inv_gamma_minus_one = equations
  let
    rho_values, rho_v1_values, rho_v2_values, rho_e_values = StructArrays.components(u_values)
    w1_values, w2_values, w3_values, w4_values = StructArrays.components(entropy_var_values)

    @turbo thread=true for i in indices(
        (rho_values, rho_v1_values, rho_v2_values, rho_e_values,
         w1_values, w2_values, w3_values, w4_values))
      rho    = rho_values[i]
      rho_v1 = rho_v1_values[i]
      rho_v2 = rho_v2_values[i]
      rho_e  = rho_e_values[i]

      # The following is basically the same code as in `cons2entropy`
      v1 = rho_v1 / rho
      v2 = rho_v2 / rho
      v_square = v1^2 + v2^2
      p = (gamma - 1) * (rho_e - 0.5 * rho * v_square)
      s = log(p) - gamma * log(rho)
      rho_p = rho / p

      w1_values[i] = (gamma - s) * inv_gamma_minus_one - 0.5 * rho_p * v_square
      w2_values[i] = rho_p * v1
      w3_values[i] = rho_p * v2
      w4_values[i] = -rho_p
    end
  end

  # "VhP" fuses the projection "P" with interpolation to volume and face quadrature "Vh"
  apply_to_each_field(mul_by!(VhP), projected_entropy_var_values, entropy_var_values)

  # The following is semantically equivalent to
  # @threaded for i in Base.OneTo(length(projected_entropy_var_values)) #eachindex(projected_entropy_var_values)
  #   entropy_projected_u_values[i] = entropy2cons(projected_entropy_var_values[i], equations)
  # end
  # but much more efficient due to explicit optimization via `@turbo` from
  # LoopVectorization.jl.
  let
    rho_values, rho_v1_values, rho_v2_values, rho_e_values = StructArrays.components(entropy_projected_u_values)
    w1_values, w2_values, w3_values, w4_values = StructArrays.components(projected_entropy_var_values)

    @turbo thread=true for i in indices(
        (rho_values, rho_v1_values, rho_v2_values, rho_e_values,
         w1_values, w2_values, w3_values, w4_values))

      # The following is basically the same code as in `entropy2cons`
      # Convert to entropy `-rho * s` used by
      # - See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
      #   [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)
      # instead of `-rho * s / (gamma - 1)`
      gamma_minus_one = gamma - 1
      w1 = gamma_minus_one * w1_values[i]
      w2 = gamma_minus_one * w2_values[i]
      w3 = gamma_minus_one * w3_values[i]
      w4 = gamma_minus_one * w4_values[i]

      # s = specific entropy, eq. (53)
      s = gamma - w1 + (w2^2 + w3^2) / (2 * w4)

      # eq. (52)
      rho_iota = (gamma_minus_one / (-w4)^gamma)^(inv_gamma_minus_one) * exp(-s * inv_gamma_minus_one)

      # eq. (51)
      rho_values[i]    = -rho_iota * w4
      rho_v1_values[i] =  rho_iota * w2
      rho_v2_values[i] =  rho_iota * w3
      rho_e_values[i]  =  rho_iota * (1 - (w2^2 + w3^2) / (2 * w4))
    end
  end
end

function entropy_projection!(cache, u::StructArray{<:SVector{5, <:Union{Float32, Float64}}},
                             mesh::VertexMappedMesh, equations::CompressibleEulerEquations3D, dg::DGMultiFluxDiff{<:Polynomial})
  rd = dg.basis
  @unpack Vq = rd
  @unpack VhP, u_values, entropy_var_values = cache
  @unpack projected_entropy_var_values, entropy_projected_u_values = cache

  apply_to_each_field(mul_by!(Vq), u_values, u)

  # The following is semantically equivalent to
  # @threaded for i in Base.OneTo(length(u_values))
  #   entropy_var_values[i] = cons2entropy(u_values[i], equations)
  # end
  # but much more efficient due to explicit optimization via `@turbo` from
  # LoopVectorization.jl.
  @unpack gamma, inv_gamma_minus_one = equations
  let
    rho_values, rho_v1_values, rho_v2_values, rho_v3_values, rho_e_values = StructArrays.components(u_values)
    w1_values, w2_values, w3_values, w4_values, w5_values = StructArrays.components(entropy_var_values)

    @turbo thread=true for i in indices(
        (rho_values, rho_v1_values, rho_v2_values, rho_v3_values, rho_e_values,
         w1_values, w2_values, w3_values, w4_values, w5_values))
      rho    = rho_values[i]
      rho_v1 = rho_v1_values[i]
      rho_v2 = rho_v2_values[i]
      rho_v3 = rho_v3_values[i]
      rho_e  = rho_e_values[i]

      # The following is basically the same code as in `cons2entropy`
      v1 = rho_v1 / rho
      v2 = rho_v2 / rho
      v3 = rho_v3 / rho
      v_square = v1^2 + v2^2 + v3^2
      p = (gamma - 1) * (rho_e - 0.5 * rho * v_square)
      s = log(p) - gamma * log(rho)
      rho_p = rho / p

      w1_values[i] = (gamma - s) * inv_gamma_minus_one - 0.5 * rho_p * v_square
      w2_values[i] = rho_p * v1
      w3_values[i] = rho_p * v2
      w4_values[i] = rho_p * v3
      w5_values[i] = -rho_p
    end
  end

  # "VhP" fuses the projection "P" with interpolation to volume and face quadrature "Vh"
  apply_to_each_field(mul_by!(VhP), projected_entropy_var_values, entropy_var_values)

  # The following is semantically equivalent to
  # @threaded for i in Base.OneTo(length(projected_entropy_var_values)) #eachindex(projected_entropy_var_values)
  #   entropy_projected_u_values[i] = entropy2cons(projected_entropy_var_values[i], equations)
  # end
  # but much more efficient due to explicit optimization via `@turbo` from
  # LoopVectorization.jl.
  let
    rho_values, rho_v1_values, rho_v2_values, rho_v3_values, rho_e_values = StructArrays.components(entropy_projected_u_values)
    w1_values, w2_values, w3_values, w4_values, w5_values = StructArrays.components(projected_entropy_var_values)

    @turbo thread=true for i in indices(
        (rho_values, rho_v1_values, rho_v2_values, rho_v3_values, rho_e_values,
         w1_values, w2_values, w3_values, w4_values, w5_values))

      # The following is basically the same code as in `entropy2cons`
      # Convert to entropy `-rho * s` used by
      # - See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
      #   [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)
      # instead of `-rho * s / (gamma - 1)`
      gamma_minus_one = gamma - 1
      w1 = gamma_minus_one * w1_values[i]
      w2 = gamma_minus_one * w2_values[i]
      w3 = gamma_minus_one * w3_values[i]
      w4 = gamma_minus_one * w4_values[i]
      w5 = gamma_minus_one * w5_values[i]

      # s = specific entropy, eq. (53)
      s = gamma - w1 + (w2^2 + w3^2 + w4^2) / (2 * w5)

      # eq. (52)
      rho_iota = (gamma_minus_one / (-w5)^gamma)^(inv_gamma_minus_one) * exp(-s * inv_gamma_minus_one)

      # eq. (51)
      rho_values[i]    = -rho_iota * w5
      rho_v1_values[i] =  rho_iota * w2
      rho_v2_values[i] =  rho_iota * w3
      rho_v3_values[i] =  rho_iota * w4
      rho_e_values[i]  =  rho_iota * (1 - (w2^2 + w3^2 + w4^2) / (2 * w5))
    end
  end
end



end # @muladd
