module TestExamples2DShallowWater

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Two-Layer Shallow Water" begin
  @trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_convergence.jl"),
    l2    = [0.00040408821395668607, 0.005467111999615421, 0.006149343132476071, 
             0.0002908715383919168, 0.0030119122689691263, 0.0026807563709877833, 
             8.87363092144442e-6],
    linf  = [0.002822049581857833, 0.01485800869399001, 0.0176001129665746, 0.0016322251417291156, 
             0.009364726422072867, 0.008414515246343512, 3.361991620143279e-5],
    tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_shallowwater_twolayer_convergence.jl with flux_es" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_convergence.jl"),
    l2    = [0.0002471488167316139, 0.0019215754323758576, 0.002383419434213394,
             0.00021257919329527273, 0.0011299806904463232, 0.0009191094806772328, 
             8.873630921447822e-6],
    linf  = [0.0016101416581815187, 0.007660451876320362, 0.009124402864519432, 
             0.001349647389053632, 0.0035576729004753727, 0.0029683897481730392, 
             3.361991620143279e-5],
    surface_flux = (flux_es, flux_nonconservative_fjordholm_etal),
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
end

end # module
