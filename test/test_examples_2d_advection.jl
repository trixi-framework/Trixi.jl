module TestExamples2DAdvection

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Linear scalar advection" begin
  @testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @testset "elixir_advection_extended.jl with polydeg=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [0.052738240907090894],
      linf = [0.08754218386076529],
      polydeg=1)
  end

  @testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [1.2148032454832534e-5],
      linf = [6.495644795245781e-5])
  end

  @testset "elixir_advection_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
      l2   = [0.0016133676309497648],
      linf = [0.013960195840607481])
  end

  @testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [0.009907208939244499],
      linf = [0.04335954152178878])
  end

  @testset "elixir_advection_amr_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
      l2   = [0.007013561257721758],
      linf = [0.039176916074623536])
  end

  @testset "elixir_advection_amr_visualization.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_visualization.jl"),
      l2   = [0.0010300631535183275],
      linf = [0.009109608720471729],
      visualization = VisualizationCallback(interval=20,
                      clims=(0,1),
                      plot_creator=Trixi.save_plot))
  end

  @testset "elixir_advection_timeintegration.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [0.005575819174358793],
      linf = [0.030180876541832102])
  end

  @testset "elixir_advection_timeintegration.jl with carpenter_kennedy_erk43" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [0.00567625639783006],
      linf = [0.02989004624813346],
      ode_algorithm=Trixi.CarpenterKennedy2N43(),
      cfl = 1.0)
  end

  @testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk94" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [0.005573040696209662],
      linf = [0.030176768981231283],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar94())
  end

  @testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk32" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [0.005563707516447738],
      linf = [0.02964401754871232],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar32(),
      cfl = 1.0)
  end

  @testset "elixir_advection_callbacks.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_callbacks.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end
end

end # module
