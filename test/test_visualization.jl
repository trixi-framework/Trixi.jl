module TestVisualization

using Test
using Documenter
using Trixi
using Plots

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# Run various visualization tests
@testset "Visualization tests" begin
  # Run Trixi
  @test_nowarn_debug trixi_include(@__MODULE__, joinpath(examples_dir(), "2d", "elixir_euler_blast_wave_amr.jl"),
                             tspan=(0,0.1))

  @testset "PlotData2D, PlotDataSeries2D, PlotMesh2D" begin
    # Constructor
    @test PlotData2D(sol) isa PlotData2D
    @test PlotData2D(sol; nvisnodes=0, grid_lines=false, solution_variables=cons2cons) isa PlotData2D
    pd = PlotData2D(sol)

    # show
    @test_nowarn_debug show(stdout, pd)
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
    @test_nowarn_debug show(stdout, pds)
    println(stdout)

    # getmesh/PlotMesh2D
    @test getmesh(pd) == Trixi.PlotMesh2D(pd)
    @test getmesh(pd).plot_data == pd
    @test_nowarn_debug show(stdout, getmesh(pd))
    println(stdout)
  end

  @testset "plot recipes" begin
    pd = PlotData2D(sol)

    @test_nowarn_debug plot(sol)
    @test_nowarn_debug plot(pd)
    @test_nowarn_debug plot(pd["p"])
    @test_nowarn_debug plot(getmesh(pd))
  end

  @testset "plot 3D" begin
    @test_nowarn_debug trixi_include(@__MODULE__, joinpath(examples_dir(), "3d", "elixir_advection_basic.jl"),
                               tspan=(0,0.1))
    @test PlotData2D(sol) isa PlotData2D
  end

  @testset "plotting TimeIntegratorSolution" begin
    @test_nowarn_debug trixi_include(@__MODULE__, joinpath(examples_dir(), "2d", "elixir_hypdiff_lax_friedrichs.jl"))
    @test_nowarn_debug plot(sol)
  end

  @testset "VisualizationCallback" begin
    # To make CI tests work, disable showing a plot window with the GR backend of the Plots package
    # Xref: https://github.com/jheinen/GR.jl/issues/278
    # Xref: https://github.com/JuliaPlots/Plots.jl/blob/8cc6d9d48755ba452a2835f9b89d3880e9945377/test/runtests.jl#L103
    if !isinteractive()
      restore = get(ENV, "GKSwstype", nothing)
      ENV["GKSwstype"] = "100"
    end

    @test_nowarn_debug trixi_include(@__MODULE__,
                               joinpath(examples_dir(), "2d", "elixir_advection_amr_visualization.jl"),
                               visualization = VisualizationCallback(interval=20,
                                               clims=(0,1),
                                               plot_creator=Trixi.save_plot),
                               tspan=(0.0, 2.0))

    @testset "elixir_advection_amr_visualization.jl with save_plot" begin
      @test isfile(joinpath(outdir, "solution_000000.png"))
      @test isfile(joinpath(outdir, "solution_000020.png"))
      @test isfile(joinpath(outdir, "solution_000024.png"))
    end

    @testset "show" begin
      @test_nowarn_debug show(stdout, visualization)
      println(stdout)

      @test_nowarn_debug show(stdout, "text/plain", visualization)
      println(stdout)
    end

    # Restore GKSwstype to previous value (if it was set)
    if !isinteractive()
      if isnothing(restore)
        delete!(ENV, "GKSwstype")
      else
        ENV["GKSwstype"] = restore
      end
    end
  end
end

end #module
