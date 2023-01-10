module TestExamples1DShallowWaterTwoLayer

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "Shallow Water Two layer" begin
  @trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_convergence.jl"),
    l2    = [0.0050681532925156945, 0.002089013899370176, 0.005105544300292713, 0.002526442122643306,
             0.0004744186597732706],
    linf  = [0.022256679217306008, 0.005421833004652266, 0.02233993939574197, 0.008765261497422516,
             0.0008992474511784199],
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_convergence.jl with flux_es" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_convergence.jl"),
    l2    = [0.0027681377074701345, 0.0018007543202559165, 0.0028036917433720576,
             0.0013980358596935737, 0.0004744186597732706], 
    linf  = [0.005699303919826093, 0.006432952918256296, 0.0058507082844360125, 0.002717615543961216,
             0.0008992474511784199],
    surface_flux=(flux_es, flux_nonconservative_fjordholm_etal),
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_well_balanced.jl"),
      l2 = [8.949288784402005e-16, 4.0636427176237915e-17, 0.001002881985401548,
             2.133351105037203e-16, 0.0010028819854016578],
     linf = [2.6229018956769323e-15, 1.878451903240623e-16, 0.005119880996670156,
             8.003199803957679e-16, 0.005119880996670666],
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_dam_break.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_dam_break.jl"),
      l2 = [0.3545579205941352, 1.6671098325838127, 0.7900767617260193, 0.9385785869258338,
             1.0023746892375034],
     linf = [0.6591615763025229, 2.0519328695833847, 1.1357745874830598, 1.2344402760174087, 1.1],
    tspan = (0.0, 0.25))
  end

end

end # module
