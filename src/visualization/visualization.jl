# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See TODO: link-to-my-blog-post
@muladd begin


include("plot_recipes.jl")
include("interpolate.jl")
include("convert.jl")
include("adapt.jl")


end # @muladd
