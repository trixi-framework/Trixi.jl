# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("positivity_zhang_shu.jl")
include("subcell_limiter_idp_correction.jl")
include("subcell_bounds_check.jl")
include("entropy_bounded_limiter.jl")
end # @muladd
