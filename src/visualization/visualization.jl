# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi/
@muladd begin


include("plot_recipes.jl")
include("interpolate.jl")
include("convert.jl")
include("adapt.jl")


end # @muladd
