module TestExamples2DShallowWaterTwoLayer

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Two-Layer Shallow Water" begin
  @trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_convergence.jl"),
    l2   = [0.0004040147445601598, 0.005466848793475609, 0.006149138398472166, 0.0002908599437447256,
          0.003011817461911792, 0.0026806180089700674, 8.873630921431545e-6],
    linf = [0.002822006686981293, 0.014859895905040332, 0.017590546190827894, 0.0016323702636176218,
            0.009361402900653015, 0.008411036357379165, 3.361991620143279e-5],
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_convergence.jl with flux_es_fjordholm_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_convergence.jl"),
    l2   = [0.00024709443131137236, 0.0019215286339769443, 0.0023833298173254447, 
          0.00021258247976270914, 0.0011299428031136195, 0.0009191313765262401,
          8.873630921431545e-6], 
    linf = [0.0016099763244645793, 0.007659242165565017, 0.009123320235427057, 
            0.0013496983982568267, 0.0035573687287770994, 0.00296823235874899,
            3.361991620143279e-5],
    surface_flux = (flux_es_fjordholm_etal, flux_nonconservative_fjordholm_etal),
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_well_balanced.jl"),
    l2   = [3.2935164267930016e-16, 4.6800825611195103e-17, 4.843057532147818e-17, 
          0.0030769233188015013, 1.4809161150389857e-16, 1.509071695038043e-16, 
          0.0030769233188014935],
    linf = [2.248201624865942e-15, 2.346382070278936e-16, 2.208565017494899e-16, 
            0.026474051138910493, 9.237568031609006e-16, 7.520758026187046e-16, 
            0.026474051138910267],
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_well_balanced with flux_lax_friedrichs.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_well_balanced.jl"),
    l2    = [2.0525741072929735e-16, 6.000589392730905e-17, 6.102759428478984e-17, 
             0.0030769233188014905, 1.8421386173122792e-16, 1.8473184927121752e-16, 
             0.0030769233188014935],
    linf  = [7.355227538141662e-16, 2.960836949170518e-16, 4.2726562436938764e-16,
             0.02647405113891016, 1.038795478061861e-15, 1.0401789378532516e-15,
             0.026474051138910267],
    surface_flux = (flux_lax_friedrichs, flux_nonconservative_fjordholm_etal),
    tspan = (0.0, 0.25))
  end
end

end # module
