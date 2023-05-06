# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# du .= zero(eltype(du)) doesn't scale when using multiple threads.
# See https://github.com/trixi-framework/Trixi.jl/pull/924 for a performance comparison.
function reset_du!(du, dg, cache)
  @threaded for element in eachelement(dg, cache)
      du[.., element] .= zero(eltype(du))
  end

  return du
end


#     pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)
#
# Given blending factors `alpha` and the solver `dg`, fill
# `element_ids_dg` with the IDs of elements using a pure DG scheme and
# `element_ids_dgfv` with the IDs of elements using a blended DG-FV scheme.
function pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg::DG, cache)
  empty!(element_ids_dg)
  empty!(element_ids_dgfv)

  for element in eachelement(dg, cache)
    # Clip blending factor for values close to zero (-> pure DG)
    dg_only = isapprox(alpha[element], 0, atol=1e-12)
    if dg_only
      push!(element_ids_dg, element)
    else
      push!(element_ids_dgfv, element)
    end
  end

  return nothing
end


function volume_jacobian(element, mesh::TreeMesh, cache)
  return inv(cache.elements.inverse_jacobian[element])^ndims(mesh)
end



# Indicators used for shock-capturing and AMR
include("indicators.jl")
include("indicators_1d.jl")
include("indicators_2d.jl")
include("indicators_3d.jl")

# Container data structures
include("containers.jl")

# 1D DG implementation
include("dg_1d.jl")
include("dg_1d_parabolic.jl")

# 2D DG implementation
include("dg_2d.jl")
include("dg_2d_parallel.jl")
include("dg_2d_parabolic.jl")

# 3D DG implementation
include("dg_3d.jl")
include("dg_3d_parabolic.jl")

# Auxiliary functions that are specialized on this solver
# as well as specialized implementations used to improve performance
include("dg_2d_compressible_euler.jl")
include("dg_3d_compressible_euler.jl")


end # @muladd
