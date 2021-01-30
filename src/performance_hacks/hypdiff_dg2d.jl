
# This file contains implementations of methods specific to the
# `HyperbolicDiffusionEquations2D` for DG methods.
# We can easily let Chris Elrod generate optimized SIMD code via LoopVectorization
# for the weak form volume integral in this case. To get the best results, we
# also need to pass more information about the statically known sizes of `du, u`
# at compile time via HybridArrays.
# However, the performance improvement is significantly less than for linear advection.
# Maybe re-ordering the memory would help, cf. https://github.com/trixi-framework/Trixi.jl/issues/88

# We cannot use this yet because of https://github.com/mateuszbaran/HybridArrays.jl/issues/39
# @inline function wrap_array(u_ode::AbstractVector, mesh::TreeMesh{2},
#                             equations::HyperbolicDiffusionEquations2D, dg::DG, cache)
#   @boundscheck begin
#     @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
#   end
#   # We would like to use
#   #   reshape(u_ode, (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
#   # but that results in
#   #   ERROR: LoadError: cannot resize array with shared data
#   # when we resize! `u_ode` during AMR.

#   # The following version is fast and allows us to `resize!(u_ode, ...)`.
#   # OBS! Remember to `GC.@preserve` temporaries such as copies of `u_ode`
#   #      and other stuff that is only used indirectly via `wrap_array` afterwards!
#   HybridArray{Tuple{nvariables(equations), nnodes(dg), nnodes(dg), Dynamic()}}(
#     unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
#                 (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache))))
# end

function calc_volume_integral!(du::AbstractArray{<:Any,4}, u,
                               nonconservative_terms::Val{false},
                               equations::HyperbolicDiffusionEquations2D,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis
  @unpack nu, inv_Tr = equations

  @threaded for element in eachelement(dg, cache)
    # @avx for j in eachnode(dg), i in eachnode(dg)
    #   dphi_ij = du[1, i, j, element]
    #   for sum_idx in eachnode(dg)
    #     dphi_ij += -nu * (derivative_dhat[i, sum_idx] * u[2, sum_idx, j, element] +
    #                       derivative_dhat[j, sum_idx] * u[3, i, sum_idx, element])
    #   end
    #   du[1, i, j, element] = dphi_ij

    #   dq1_ij = du[2, i, j, element]
    #   for sum_idx in eachnode(dg)
    #     dq1_ij += -inv_Tr * (derivative_dhat[i, sum_idx] * u[1, sum_idx, j, element])
    #   end
    #   du[2, i, j, element] = dq1_ij

    #   dq2_ij = du[3, i, j, element]
    #   for sum_idx in eachnode(dg)
    #     dq2_ij += -inv_Tr * (derivative_dhat[j, sum_idx] * u[1, i, sum_idx, element])
    #   end
    #   du[3, i, j, element] = dq2_ij
    # end

    # @avx for j in eachnode(dg), i in eachnode(dg)
    #   dphi_ij = du[1, i, j, element]
    #   dq1_ij  = du[2, i, j, element]
    #   dq2_ij  = du[3, i, j, element]
    #   for sum_idx in eachnode(dg)
    #     dphi_ij += -nu * (derivative_dhat[i, sum_idx] * u[2, sum_idx, j, element] +
    #                       derivative_dhat[j, sum_idx] * u[3, i, sum_idx, element])
    #     dq1_ij += -inv_Tr * (derivative_dhat[i, sum_idx] * u[1, sum_idx, j, element])
    #     dq2_ij += -inv_Tr * (derivative_dhat[j, sum_idx] * u[1, i, sum_idx, element])
    #   end
    #   du[1, i, j, element] = dphi_ij
    #   du[2, i, j, element] = dq1_ij
    #   du[3, i, j, element] = dq2_ij
    # end

    @avx for j in eachnode(dg), i in eachnode(dg)
      dphi_ij = zero(eltype(du))
      dq1_ij  = zero(eltype(du))
      dq2_ij  = zero(eltype(du))
      for sum_idx in eachnode(dg)
        dphi_ij += (derivative_dhat[i, sum_idx] * u[2, sum_idx, j, element] +
                    derivative_dhat[j, sum_idx] * u[3, i, sum_idx, element])
        dq1_ij  += (derivative_dhat[i, sum_idx] * u[1, sum_idx, j, element])
        dq2_ij  += (derivative_dhat[j, sum_idx] * u[1, i, sum_idx, element])
      end
      du[1, i, j, element] -= nu * dphi_ij
      du[2, i, j, element] -= inv_Tr * dq1_ij
      du[3, i, j, element] -= inv_Tr * dq1_ij
    end

    # @avx for j in eachnode(dg), i in eachnode(dg)
    #   dphi_ij = du[1, i, j, element]
    #   for sum_idx in eachnode(dg)
    #     dphi_ij += -nu * (derivative_dhat[i, sum_idx] * u[2, sum_idx, j, element] +
    #                       derivative_dhat[j, sum_idx] * u[3, i, sum_idx, element])
    #   end
    #   du[1, i, j, element] = dphi_ij
    # end
    # @avx for j in eachnode(dg), i in eachnode(dg)
    #   dq1_ij = du[2, i, j, element]
    #   for sum_idx in eachnode(dg)
    #     dq1_ij += -inv_Tr * (derivative_dhat[i, sum_idx] * u[1, sum_idx, j, element])
    #   end
    #   du[2, i, j, element] = dq1_ij
    # end
    # @avx for j in eachnode(dg), i in eachnode(dg)
    #   dq2_ij = du[3, i, j, element]
    #   for sum_idx in eachnode(dg)
    #     dq2_ij += -inv_Tr * (derivative_dhat[j, sum_idx] * u[1, i, sum_idx, element])
    #   end
    #   du[3, i, j, element] = dq2_ij
    # end
  end

  return nothing
end

