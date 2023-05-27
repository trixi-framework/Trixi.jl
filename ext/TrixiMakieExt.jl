# Package extension for adding Makie-based features to Trixi.jl
module TrixiMakieExt

# Required for visualization code
using Makie: Makie, GeometryBasics

# Use all exported symbols to avoid having to rewrite `recipes_makie.jl`
using Trixi

# Use additional symbols that are not exported
using Trixi: PlotData2DTriangulated, TrixiODESolution, PlotDataSeries, ScalarData, @muladd

# Import functions such that they can be extended with new methods
import Trixi: iplot, iplot!

include("../src/visualization/recipes_makie.jl")

end
