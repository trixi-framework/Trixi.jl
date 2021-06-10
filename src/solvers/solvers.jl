# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See TODO: link-to-my-blog-post
@muladd begin


include("dg_tree/dg.jl")
include("dg_curved/dg.jl")
include("dg_unstructured_quad/dg.jl")
include("dg_p4est/dg.jl")
include("dg_common.jl")
include("fdsbp_tree/fdsbp_2d.jl")


end # @muladd
