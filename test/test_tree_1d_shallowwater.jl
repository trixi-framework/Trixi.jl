module TestExamples1DShallowWater

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")#joinpath(examples_dir(), "tree_1d_dgsem")

@testset "Shallow Water" begin
  @trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
      l2   = [1.10855390, 4.75051966e-01, 5.22183534e-01],
      linf = [1.84147098, 7.52800000e-01, 8.41470985e-01],
      tspan = (0.0, 2.0))
  end

  @trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
      l2   = [1.36330288, 3.70601713e-12, 1.36330288],
      linf = [1.61891148, 3.72525608e-12, 1.61891148],
      tspan = (0.0, 100.0))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [1.72639378e-03, 1.51395693e-02, 4.43649173e-05],
      linf = [6.63829803e-03, 4.74841543e-02, 9.09837978e-05],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [1.73837899e-03, 1.51869435e-02, 4.43649173e-05],
      linf = [6.16810476e-03, 4.50883562e-02, 9.09837978e-05],
      tspan = (0.0, 1.0), surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal))
  end
end

end # module
