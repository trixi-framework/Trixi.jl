module TrixiPlotsExt

# Load the required packages
using Plots: Plots
using Trixi
using Trixi: AbstractPlotData, PlotDataSeries, PlotMesh, PlotData1D, PlotData2D,
             ScalarPlotData2D, PlotData2DCartesian, PlotData2DTriangulated, getmesh,
             TrixiODESolution, AbstractSemidiscretization, ScalarData, DiscreteCallback,
             CallbackSet, ODEProblem, ODESolution
using MuladdMacro: @muladd
using RecipesBase

@muladd begin
#! format: noindent

# Include the file that actually has the plotting logic
include("../src/visualization/recipes_plots.jl")
end # @muladd

end # module TrixiPlotsExt
