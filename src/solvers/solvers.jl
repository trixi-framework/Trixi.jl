# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi/
@muladd begin


include_fast("dg_tree/dg.jl")
include_fast("dg_curved/dg.jl")
include_fast("dg_unstructured_quad/dg.jl")
include_fast("dg_p4est/dg.jl")
include_fast("dg_common.jl")
include_fast("fdsbp_tree/fdsbp_2d.jl")


end # @muladd
