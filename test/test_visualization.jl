module TestVisualization

using Test
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
  # Run 2D tests with elixirs for both mesh types
  test_examples_2d = Dict(
    "TreeMesh" => "elixir_euler_blast_wave_amr.jl",
    "CurvedMesh" => "elixir_euler_source_terms_waving_flag.jl",
    "UnstructuredQuadMesh" => "elixir_euler_unstructured_quad_wall_bc.jl"
  )

  @testset "PlotData2D, PlotDataSeries2D, PlotMesh2D with $mesh" for mesh in keys(test_examples_2d)
    # Run Trixi
    @test_nowarn_debug trixi_include(@__MODULE__, joinpath(examples_dir(), "2d", test_examples_2d[mesh]),
                                     tspan=(0,0.1))

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

    @testset "2D plot recipes" begin
      pd = PlotData2D(sol)

      @test_nowarn_debug plot(sol)
      @test_nowarn_debug plot(pd)
      @test_nowarn_debug plot(pd["p"])
      @test_nowarn_debug plot(getmesh(pd))
    end

    @testset "1D plot from 2D solution" begin
      @testset "Create 1D plot as slice" begin
        @test_nowarn_debug PlotData1D(sol, slice=:y, point=(-0.5, 0.0)) isa PlotData1D
        pd1D = PlotData1D(sol, slice=:y, point=(-0.5, 0.0))
        @test_nowarn_debug plot(pd1D)
      end

      if mesh == "TreeMesh"
        @testset "Create 1D plot along curve" begin
          curve = zeros(2,10)
          curve[1,:] = range(-1,-0.5,length=10)
          @test_nowarn_debug PlotData1D(sol, curve=curve) isa PlotData1D
          pd1D = PlotData1D(sol, curve=curve)
          @test_nowarn_debug plot(pd1D)
        end
      end
    end
  end

  @testset "PlotData1D, PlotDataSeries1D, PlotMesh1D" begin
    # Run Trixi
    @test_nowarn_debug trixi_include(@__MODULE__, joinpath(examples_dir(), "1d", "elixir_euler_blast_wave.jl"),
                                     tspan=(0,0.1))

    # Constructor
    @test PlotData1D(sol) isa PlotData1D
    pd = PlotData1D(sol)

    # show
    @test_nowarn_debug show(stdout, pd)
    println(stdout)

    # getindex
    @test pd["rho"] == Trixi.PlotDataSeries1D(pd, 1)
    @test pd["v1"] == Trixi.PlotDataSeries1D(pd, 2)
    @test pd["p"] == Trixi.PlotDataSeries1D(pd, 3)
    @test_throws KeyError pd["does not exist"]

    # convenience methods for mimicking a dictionary
    @test pd[begin] == Trixi.PlotDataSeries1D(pd, 1)
    @test pd[end] == Trixi.PlotDataSeries1D(pd, 3)
    @test length(pd) == 3
    @test size(pd) == (3,)
    @test keys(pd) == ("rho", "v1", "p")
    @test eltype(pd) == Pair{String, Trixi.PlotDataSeries1D}
    @test [v for v in pd] == ["rho" => Trixi.PlotDataSeries1D(pd, 1),
                              "v1" => Trixi.PlotDataSeries1D(pd, 2),
                              "p" => Trixi.PlotDataSeries1D(pd, 3)]

    # PlotDataSeries1D
    pds = pd["p"]
    @test pds.plot_data == pd
    @test pds.variable_id == 3
    @test_nowarn_debug show(stdout, pds)
    println(stdout)

    # getmesh/PlotMesh1D
    @test getmesh(pd) == Trixi.PlotMesh1D(pd)
    @test getmesh(pd).plot_data == pd
    @test_nowarn_debug show(stdout, getmesh(pd))
    println(stdout)

    # nvisnodes
    @test size(pd.data) == (512, 3)
    pd0 = PlotData1D(sol, nvisnodes=0)
    @test size(pd0.data) == (256, 3)
    pd2 = PlotData1D(sol, nvisnodes=2)
    @test size(pd2.data) == (128, 3)

    @testset "1D plot recipes" begin
      pd = PlotData1D(sol)

      @test_nowarn_debug plot(sol)
      @test_nowarn_debug plot(pd)
      @test_nowarn_debug plot(pd["p"])
      @test_nowarn_debug plot(getmesh(pd))
    end

    # Fake a PlotDataXD objects to test code for plotting multiple variables on at least two rows
    # with at least one plot remaining empty
    @testset "plotting multiple variables" begin
      x = collect(0.0:0.1:1.0)
      data1d = rand(5, 11)
      variable_names = string.('a':'e')
      mesh_vertices_x1d = [x[begin], x[end]]
      fake1d = PlotData1D(x, data1d, variable_names, mesh_vertices_x1d, 0)
      @test_nowarn_debug plot(fake1d)

      y = x
      data2d = [rand(11,11) for _ in 1:5]
      mesh_vertices_x2d = [0.0, 1.0, 1.0, 0.0]
      mesh_vertices_y2d = [0.0, 0.0, 1.0, 1.0]
      fake2d = PlotData2D(x, y, data2d, variable_names, mesh_vertices_x2d, mesh_vertices_y2d, 0, 0)
      @test_nowarn_debug plot(fake2d)
    end
  end

  @testset "plot time series" begin
    @test_nowarn_debug trixi_include(@__MODULE__,
                                     joinpath(examples_dir(), "2d", "elixir_ape_gaussian_source.jl"),
                                     tspan=(0, 0.05))

    @test_nowarn_debug plot(time_series, 1)
    @test PlotData1D(time_series, 1) isa PlotData1D
  end

  @testset "adapt_to_mesh_level" begin
    @test_nowarn_debug trixi_include(@__MODULE__, joinpath(examples_dir(), "2d", "elixir_advection_basic.jl"),
                                     tspan=(0,0.1))
    @test adapt_to_mesh_level(sol, 5) isa Tuple

    u_ode_level5, semi_level5 = adapt_to_mesh_level(sol, 5)
    u_ode_level4, semi_level4 = adapt_to_mesh_level(u_ode_level5, semi_level5, 4)
    @test isapprox(sol.u[end], u_ode_level4, atol=1e-13)

    @test adapt_to_mesh_level!(sol, 5) isa Tuple
    @test isapprox(sol.u[end], u_ode_level5, atol=1e-13)
  end

  @testset "plot 3D" begin
    @test_nowarn_debug trixi_include(@__MODULE__, joinpath(examples_dir(), "3d", "elixir_advection_basic.jl"),
                                     tspan=(0,0.1))
    @test PlotData2D(sol) isa PlotData2D

    @testset "1D plot from 3D solution" begin
      @testset "Create 1D plot as slice" begin
        @test_nowarn_debug PlotData1D(sol) isa PlotData1D
        pd1D = PlotData1D(sol)
        @test_nowarn_debug plot(pd1D)
      end

      @testset "Create 1D plot along curve" begin
        curve = zeros(3,10)
        curve[1,:] = range(-1,-0.5,length=10)
        @test_nowarn_debug PlotData1D(sol, curve=curve) isa PlotData1D
        pd1D = PlotData1D(sol, curve=curve)
        @test_nowarn_debug plot(pd1D)
      end
    end
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
