module TestExamples1DTwoLayerShallowWater

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "Shallow Water Two layer" begin
  @trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_convergence.jl"),
    (l2  = [0.04673980168817153, 0.04763185697767758, 0.03533867539601806, 0.031180099635701033,
            0.0004744186597732739],
    linf = [0.15115950324422744, 0.18188014817018994, 0.11367746254573174, 0.0572968267760412,
            0.0008992474511784199],
    tspan = (0.0, 0.25)))
  end

  @trixi_testset "elixir_shallowwater_twolayer_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_well_balanced.jl"),
    (l2  = [0.0001043472444563723, 2.608364617470423e-8, 4.3922341148122115e-8, 8.96174409911765e-9,
            0.00010431282550286695],
    linf = [0.0008753595197996122, 2.0606704571879408e-7, 3.199969511156759e-7,
            6.509989418545897e-8, 0.0008750044571282074],
    tspan = (0.0, 0.25)))
  end

  @trixi_testset "elixir_shallowwater_twolayer_dam_break.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_twolayer_dam_break.jl"),
    (l2  = [0.7901543619118623, 0.3547187997797987, 0.943203493765727, 1.6706539965065055,
            1.0023823643770289],
    linf = [1.1442506604847242, 0.6413253158843664, 1.2779703891416647, 2.1214560802803004, 1.1],
    tspan = (0.0, 0.25)))
  end

end

end # module
