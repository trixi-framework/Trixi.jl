
# This file contains implementations of methods specific to the
# `LinearScalarAdvectionEquation2D` for DG methods.
# We can easily let Chris Elrod generate optimized SIMD code via LoopVectorization
# for the weak form volume integral in this case. To get the best results, we
# also need to pass more information about the statically known sizes of `du, u`
# at compile time via HybridArrays.

@inline function wrap_array(u_ode::AbstractVector, mesh::TreeMesh{2},
                            equations::LinearScalarAdvectionEquation2D, dg::DG, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  # We would like to use
  #   reshape(u_ode, (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache)))
  # but that results in
  #   ERROR: LoadError: cannot resize array with shared data
  # when we resize! `u_ode` during AMR.

  # The following version is fast and allows us to `resize!(u_ode, ...)`.
  # OBS! Remember to `GC.@preserve` temporaries such as copies of `u_ode`
  #      and other stuff that is only used indirectly via `wrap_array` afterwards!
  HybridArray{Tuple{nvariables(equations), nnodes(dg), nnodes(dg), Dynamic()}}(
    unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
                (nvariables(equations), nnodes(dg), nnodes(dg), nelements(dg, cache))))
end

function calc_volume_integral!(du::AbstractArray{<:Any,4}, u,
                               nonconservative_terms::Val{false},
                               equations::LinearScalarAdvectionEquation2D,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis
  a1, a2 = equations.advectionvelocity

  @threaded for element in eachelement(dg, cache)
    @avx for j in eachnode(dg), i in eachnode(dg)
      duij = du[1, i, j, element]
      for sum_idx in eachnode(dg)
        duij += (derivative_dhat[i, sum_idx] * a1 * u[1, sum_idx, j, element] +
                 derivative_dhat[j, sum_idx] * a2 * u[1, i, sum_idx, element])
      end
      du[1, i, j, element] = duij
    end
  end

  return nothing
end

