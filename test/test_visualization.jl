module TestVisualization

using Test
using Documenter
using Trixi
using RecipesBase

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# Run various visualization tests
@testset "Visualization tests" begin
  # Run Trixi
  @test_nowarn trixi_include(joinpath(examples_dir(), "2d", "elixir_euler_blast_wave_amr.jl"),
                             tspan=(0,0.1))

  @testset "PlotData2D, PlotDataSeries2D, PlotMesh2D" begin
    # Constructor
    @test PlotData2D(sol) isa PlotData2D
    pd = PlotData2D(sol)

    # show
    @test_nowarn show(stdout, pd)
    println(stdout)

    # getindex
    @test pd["rho"] == PlotDataSeries2D(pd, 1)
    @test pd["v1"] == PlotDataSeries2D(pd, 2)
    @test pd["v2"] == PlotDataSeries2D(pd, 3)
    @test pd["p"] == PlotDataSeries2D(pd, 4)
    @test_throws KeyError pd["does not exist"]

    # convenience methods for mimicking a dictionary
    @test pd[begin] == "rho"
    @test pd[end] == "p"
    @test length(pd) == 4
    @test size(pd) == (4,)
    @test keys(pd) == ("rho", "v1", "v2", "p")
    @test eltype(pd) == Pair{String, PlotDataSeries2D}
    @test [v for v in pd] == ["rho", "v1", "v2", "p"]

    # PlotDataSeries2D
    pds = pd["p"]
    @test pds.plot_data == pd
    @test pds.variable_id == 4
    @test_nowarn show(stdout, pds)
    println(stdout)

    # getmesh/PlotMesh2D
    @test getmesh(pd) == PlotMesh2D(pd)
    @test getmesh(pd).plot_data == pd
    @test_nowarn show(stdout, getmesh(pd))
    println(stdout)
  end

  @testset "plot recipes" begin
    pd = PlotData2D(sol)

    @test RecipesBase.apply_recipe(Dict{Symbol,Any}(), sol) isa Vector{RecipesBase.Recipedata}
    @test RecipesBase.apply_recipe(Dict{Symbol,Any}(), pd) isa Vector{RecipesBase.Recipedata}
    @test RecipesBase.apply_recipe(Dict{Symbol,Any}(), pd["p"]) isa Vector{RecipesBase.Recipedata}
    @test RecipesBase.apply_recipe(Dict{Symbol,Any}(), getmesh(pd)) isa Vector{RecipesBase.Recipedata}
  end

  @testset "plot 3D" begin
    @test_nowarn trixi_include(joinpath(examples_dir(), "3d", "elixir_advection_basic.jl"),
                              tspan=(0,0.1))
    @test PlotData2D(sol) isa PlotData2D
  end
end

end #module
