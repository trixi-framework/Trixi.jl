# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("positivity_zhang_shu.jl")
include("subcell_limiter_idp_correction.jl")
# TODO: TrixiShallowWater: move specific limiter file
include("positivity_shallow_water.jl")
end # @muladd
