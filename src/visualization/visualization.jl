# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

include("types.jl")
include("utilities.jl")
include("recipes_plots.jl")

# Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
# We do not check `isdefined(Base, :get_extension)` since Julia v1.9.0
# does not load package extensions when their dependency is loaded from
# the main environment
@static if VERSION >= v"1.9.1"
  # Add function definitions here such that they can be exported from Trixi.jl and extended in the
  # TrixiMakieExt package extension
  function iplot end
  function iplot! end
end

end # @muladd
