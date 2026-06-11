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
[GLMakie.jl](https://github.com/JuliaPlots/GLMakie.jl/). This requires
GLMakie.jl to be loaded.

The plot can be rotated (click and hold), zoomed (scroll), and panned (right click and drag).
Two toggle buttons control whether mesh lines are visible above and below the solution surface.

See also [`iplot!`](@ref) for adding an interactive surface plot to an existing Makie.jl figure or axis.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function iplot end
"""
    iplot!(fig_axis, pd; kwargs...)

Add an interactive surface plot of the scalar data `pd` to an existing Makie.jl figure or axis
object `fig_axis`. Requires [GLMakie.jl](https://github.com/JuliaPlots/GLMakie.jl/) to be loaded.

See also [`iplot`](@ref) for creating a new interactive surface plot.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function iplot! end
"""
    trixiheatmap(pd; kwargs...)

Plot a `PlotDataSeries` from a `PlotData2DTriangulated` object as a 2D heatmap
in a new figure.
Requires a Makie.jl backend such as [CairoMakie.jl](https://github.com/JuliaPlots/CairoMakie.jl) to be loaded.

Note: For Cartesian mesh type ([`TreeMesh`](@ref)), use Makie.jl's built-in `heatmap`
instead, as `trixiheatmap` only supports triangulated data.

See also [`trixiheatmap!`](@ref) for adding a heatmap to an existing Makie.jl axis.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function trixiheatmap end
"""
    trixiheatmap!(ax, pd; kwargs...)

Add a heatmap of a `PlotDataSeries` from a `PlotData2DTriangulated` object to an existing
Makie.jl axis `ax`.
Requires a Makie.jl backend such as [CairoMakie.jl](https://github.com/JuliaPlots/CairoMakie.jl) to be loaded.

Note: For Cartesian mesh type ([`TreeMesh`](@ref)), use Makie.jl's built-in `heatmap!`
instead, as `trixiheatmap!` only supports triangulated data.

See also [`trixiheatmap`](@ref) for creating a new heatmap figure.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function trixiheatmap! end
end # @muladd
