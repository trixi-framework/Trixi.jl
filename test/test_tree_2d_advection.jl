module TestExamples2DAdvection

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Linear scalar advection" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @trixi_testset "elixir_advection_extended.jl with polydeg=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [0.052738240907090894],
      linf = [0.08754218386076529],
      polydeg=1)
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [1.2148032454832534e-5],
      linf = [6.495644795245781e-5])
  end

  @trixi_testset "elixir_advection_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
      l2   = [0.0016133676309497648],
      linf = [0.013960195840607481])
  end

  @trixi_testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [8.553422725808076e-5],
      linf = [0.0007746920671942853])
  end

  @trixi_testset "elixir_advection_amr_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
      l2   = [3.5536338520440995e-5],
      linf = [0.0005514506116067221])
  end

  @trixi_testset "elixir_advection_amr_solution_independent.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_solution_independent.jl"),
      l2   = [7.944846759690297e-5],
      linf = [0.0007454287896710293])
  end

  @trixi_testset "elixir_advection_amr_visualization.jl" begin
    # To make CI tests work, disable showing a plot window with the GR backend of the Plots package
    # Xref: https://github.com/jheinen/GR.jl/issues/278
    # Xref: https://github.com/JuliaPlots/Plots.jl/blob/8cc6d9d48755ba452a2835f9b89d3880e9945377/test/runtests.jl#L103
    if !isinteractive()
      restore = get(ENV, "GKSwstype", nothing)
      ENV["GKSwstype"] = "100"
    end

    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_visualization.jl"),
      l2   = [0.0010300631535183275],
      linf = [0.009109608720471729])

    # Restore GKSwstype to previous value (if it was set)
    if !isinteractive()
      if isnothing(restore)
        delete!(ENV, "GKSwstype")
      else
        ENV["GKSwstype"] = restore
      end
    end
  end

  @trixi_testset "elixir_advection_timeintegration.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [2.8777424975442084e-5],
      linf = [0.00047677279220256774])
  end

  @trixi_testset "elixir_advection_timeintegration.jl with carpenter_kennedy_erk43" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [2.9389967058539404e-5],
      linf = [0.0005734205598502823],
      ode_algorithm=Trixi.CarpenterKennedy2N43(),
      cfl = 1.0)
  end

  @trixi_testset "elixir_advection_timeintegration.jl with carpenter_kennedy_erk43 with maxiters=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [1.2046995151616365e-5],
      linf = [9.814863096402338e-5],
      ode_algorithm=Trixi.CarpenterKennedy2N43(),
      cfl = 1.0,
      maxiters = 1)
  end

  @trixi_testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk94" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [2.8778180933565268e-5],
      linf = [0.0004768629208851405],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar94())
  end

  @trixi_testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk32" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [4.9135265795604245e-5],
      linf = [0.0005480422381496791],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar32(),
      cfl = 1.0)
  end

  @trixi_testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk32 with maxiters=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [1.2074110061802306e-5],
      linf = [9.826959330024553e-5],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar32(),
      cfl = 1.0,
      maxiters = 1)
  end

  @trixi_testset "elixir_advection_callbacks.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_callbacks.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end
end

# Coverage test for all initial conditions
@testset "Linear scalar advection: Tests for initial conditions" begin
  # Linear scalar advection
  @trixi_testset "elixir_advection_extended.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [3.2933000250376106e-16],
      linf = [6.661338147750939e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end
end

end # module
