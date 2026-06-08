# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

include("types.jl")
include("utilities.jl")
include("utilities_p4est_t8code.jl")
include("recipes_plots.jl")

# Add function definitions here such that they can be exported from Trixi.jl and extended in the
# TrixiMakieExt package extension or by the Makie-specific code loaded by Requires.jl
"""
    iplot(sol; kwargs...)
    iplot(u, semi; kwargs...)

Create an interactive surface plot of a Trixi.jl simulation using
[GLMakie.jl](https://github.com/JuliaPlots/GLMakie.jl/). The plot can be rotated
(click and hold), zoomed (scroll), and panned (right click and drag). Two toggle buttons
control whether mesh lines are visible above and below the solution surface.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function iplot end
function iplot! end
function trixiheatmap end
function trixiheatmap! end
end # @muladd
