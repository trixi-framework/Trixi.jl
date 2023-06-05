module TestExamples1DShallowWaterTwoLayer

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Shallow Water Two layer" begin
  @trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_convergence.jl"),
    l2    = [0.005012009872109003, 0.002091035326731071, 0.005049271397924551,
             0.0024633066562966574, 0.0004744186597732739], 
    linf  = [0.0213772149343594, 0.005385752427290447, 0.02175023787351349, 
             0.008212004668840978, 0.0008992474511784199],
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_well_balanced.jl"),
    l2    = [8.949288784402005e-16, 4.0636427176237915e-17, 0.001002881985401548,
             2.133351105037203e-16, 0.0010028819854016578],
    linf  = [2.6229018956769323e-15, 1.878451903240623e-16, 0.005119880996670156,
             8.003199803957679e-16, 0.005119880996670666],
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_dam_break.jl with flux_lax_friedrichs" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_dam_break.jl"),
    l2    = [0.35490300089633237, 1.6715035180645277, 0.6960151891515254,
             0.9351487046099882, 0.7938172946965545],
    linf  = [0.6417505887480972, 1.9743522856822249, 1.1357745874830805,
             1.2361800390346178, 1.1],
    surface_flux = (flux_lax_friedrichs, flux_nonconservative_ersing_etal),
    tspan = (0.0, 0.25))
  end

end

end # module
