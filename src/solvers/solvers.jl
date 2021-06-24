# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


include("dg_tree/dg.jl")
include("dg_curved/dg.jl")
include("dg_unstructured_quad/dg.jl")
include("dg_p4est/dg.jl")
include("dg_common.jl")
include("fdsbp_tree/fdsbp_2d.jl")


end # @muladd
