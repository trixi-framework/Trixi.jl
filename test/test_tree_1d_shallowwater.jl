module TestExamples1DShallowWater

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "Shallow Water" begin
  @trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
      l2   = [0.8122354510732459, 1.01586214815876, 0.43404255061704217],
      linf = [1.4883285368551107, 3.8717508164234276, 1.7711213427919539],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
      l2   = [1.2427984842961743, 1.0332499675061871e-14, 1.2427984842961741],
      linf = [1.619041478244762, 1.266865149831811e-14, 1.6190414782447629],
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

  @trixi_testset "elixir_shallowwater_source_terms_dirichlet.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms_dirichlet.jl"),
      l2   = [0.0022851099219788917, 0.01560453773635554, 4.43649172558535e-5],
      linf = [0.008934615705174398, 0.059403169140869405, 9.098379777405796e-5],
      tspan = (0.0, 0.025))
  end

  @trixi_testset "elixir_shallowwater_well_balanced_nonperiodic.jl with dirichlet boundary" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced_nonperiodic.jl"),
      l2   = [1.725964362045055e-8, 5.0427180314307505e-16, 1.7259643530442137e-8],
      linf = [3.844551077492042e-8, 3.469453422316143e-15, 3.844551077492042e-8],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_well_nonperiodic.jl with wall boundary" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced_nonperiodic.jl"),
      l2   = [1.7259643614361866e-8, 3.5519018243195145e-16, 1.7259643530442137e-8],
      linf = [3.844551010878661e-8, 9.846474508971374e-16, 3.844551077492042e-8],
      tspan = (0.0, 0.25),
      boundary_condition = boundary_condition_slip_wall)
  end

  @trixi_testset "elixir_shallowwater_shock_capturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_shock_capturing.jl"),
      l2   = [0.28840248189190687, 0.5252262013521147, 0.2890348477852955],
      linf = [0.7565706154863674, 2.076621603471688, 0.8646939843534258],
      tspan = (0.0, 0.05))
  end
end

end # module
