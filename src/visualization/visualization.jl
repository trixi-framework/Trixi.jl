# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("types.jl")
include("utilities.jl")
include("recipes_plots.jl")

# Add function definitions here such that they can be exported from Trixi.jl and extended in the
# TrixiMakieExt package extension or by the Makie-specific code loaded by Requires.jl
function iplot end
function iplot! end
end # @muladd
