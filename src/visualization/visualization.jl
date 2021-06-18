# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi/
@muladd begin


include_fast("plot_recipes.jl")
include_fast("interpolate.jl")
include_fast("convert.jl")
include_fast("adapt.jl")


end # @muladd
