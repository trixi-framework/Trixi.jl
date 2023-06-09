# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


include("positivity_zhang_shu.jl")
include("a_posteriori_limiter.jl")
include("bounds_check.jl")


end # @muladd
