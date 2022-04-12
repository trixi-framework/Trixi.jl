module TestExamples1DShallowWater

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")#joinpath(examples_dir(), "tree_1d_dgsem")

@testset "Shallow Water" begin
  @trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
      l2   = [0.7916241311980453, 0.4101269104970603, 4.279166584958072e-7],
      linf = [1.0000007386737142, 0.7062385748682828, 7.38673714334448e-7],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
      l2   = [1.3633028849754976, 1.0332499675061871e-14, 1.3633028849754982],
      linf = [1.6189114814413754, 1.266865149831811e-14, 1.6189114814413768],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [0.0022363707373868713, 0.01576799981934617, 4.436491725585346e-5],
      linf = [0.00893601803417754, 0.05939797350246456, 9.098379777405796e-5],
      tspan = (0.0, 0.025))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [0.0022758146627220154, 0.015864082886204556, 4.436491725585346e-5],
      linf = [0.008457195427364006, 0.057201667446161064, 9.098379777405796e-5],
      tspan = (0.0, 0.025), surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal))
  end
end

end # module
