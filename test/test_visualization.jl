module TestVisualization

using Test
using Documenter
using Trixi
using Plots

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# Run various visualization tests
@testset "Visualization tests" begin
  # Run Trixi
  @test_nowarn trixi_include(@__MODULE__, joinpath(examples_dir(), "2d", "elixir_euler_blast_wave_amr.jl"),
                             tspan=(0,0.1))

  @testset "PlotData2D, PlotDataSeries2D, PlotMesh2D" begin
    # Constructor
    @test PlotData2D(sol) isa PlotData2D
    pd = PlotData2D(sol)

    # show
    @test_nowarn show(stdout, pd)
    println(stdout)

    # getindex
    @test pd["rho"] == Trixi.PlotDataSeries2D(pd, 1)
    @test pd["v1"] == Trixi.PlotDataSeries2D(pd, 2)
    @test pd["v2"] == Trixi.PlotDataSeries2D(pd, 3)
    @test pd["p"] == Trixi.PlotDataSeries2D(pd, 4)
    @test_throws KeyError pd["does not exist"]

    # convenience methods for mimicking a dictionary
    @test pd[begin] == Trixi.PlotDataSeries2D(pd, 1)
    @test pd[end] == Trixi.PlotDataSeries2D(pd, 4)
    @test length(pd) == 4
    @test size(pd) == (4,)
    @test keys(pd) == ("rho", "v1", "v2", "p")
    @test eltype(pd) == Pair{String, Trixi.PlotDataSeries2D}
    @test [v for v in pd] == ["rho" => Trixi.PlotDataSeries2D(pd, 1),
                              "v1" => Trixi.PlotDataSeries2D(pd, 2),
                              "v2" => Trixi.PlotDataSeries2D(pd, 3),
                              "p" => Trixi.PlotDataSeries2D(pd, 4)]

    # PlotDataSeries2D
    pds = pd["p"]
    @test pds.plot_data == pd
    @test pds.variable_id == 4
    @test_nowarn show(stdout, pds)
    println(stdout)

    # getmesh/PlotMesh2D
    @test getmesh(pd) == Trixi.PlotMesh2D(pd)
    @test getmesh(pd).plot_data == pd
    @test_nowarn show(stdout, getmesh(pd))
    println(stdout)
  end

  @testset "plot recipes" begin
    pd = PlotData2D(sol)

    @test_nowarn plot(sol);
    @test_nowarn plot(pd);
    @test_nowarn plot(pd["p"]);
    @test_nowarn plot(getmesh(pd));
  end

  @testset "plot 3D" begin
    @test_nowarn trixi_include(@__MODULE__, joinpath(examples_dir(), "3d", "elixir_advection_basic.jl"),
                               tspan=(0,0.1))
    @test PlotData2D(sol) isa PlotData2D
  end
end

end #module
