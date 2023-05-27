# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

include("types.jl")
include("utilities.jl")
include("recipes_plots.jl")

# Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
@static if VERSION >= v"1.10.0-DEV.1288" || VERSION >= v"1.9.1"
  # Add function definitions here such that they can be exported from Trixi.jl and extended in the
  # TrixiMakieExt package extension
  function iplot end
  function iplot! end
end

end # @muladd
