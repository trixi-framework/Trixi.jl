module TestExamples3DAdvection

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_3d_dgsem")

@testset "Linear scalar advection" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.00015975755208652597],
      linf = [0.0015038732976652147])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [0.0001780001287314664],
      linf = [0.0014520752637396939])
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_sin" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [0.002727293067654415],
      linf = [0.024833049753677727],
      initial_condition=Trixi.initial_condition_sin)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [7.033186506921888e-16],
      linf = [2.6645352591003757e-15],
      initial_condition=initial_condition_constant)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_linear_z and periodicity=false" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [5.887699520794518e-16],
      linf = [6.217248937900877e-15],
      initial_condition=Trixi.initial_condition_linear_z,
      boundary_conditions=Trixi.boundary_condition_linear_z, periodicity=false)
  end

  @trixi_testset "elixir_advection_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
      l2   = [0.0018461529502663268],
      linf = [0.01785420966285467])
  end

  @trixi_testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [9.773852895157622e-6],
      linf = [0.0005853874124926162])
  end
end

end # module
