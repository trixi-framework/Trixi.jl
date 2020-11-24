module TestExamples1DMHD

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

@testset "MHD" begin

  @testset "elixir_mhd_alfven.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven.jl"),
      l2   = [1.440611823425164e-15, 1.1373567770134494e-14, 3.024482376149653e-15, 2.0553143516814395e-15, 3.9938347410210535e-14, 3.984545392098788e-16, 2.4782402104201577e-15, 1.551737464879987e-15],
      linf = [1.9984014443252818e-15, 1.3405943022348765e-14, 3.3584246494910985e-15, 3.164135620181696e-15, 7.815970093361102e-14, 8.881784197001252e-16, 2.886579864025407e-15, 2.942091015256665e-15],
      initial_condition = initial_condition_constant,
      tspan = (0.0,1.0))
  end

  @testset "elixir_mhd_alfven.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven.jl"),
      l2   = [1.4717982305884098e-5, 1.182263100132273e-5, 2.3571203951257584e-5, 2.3571203951305505e-5, 1.8950292712340346e-6, 1.196322416573199e-16, 2.3661336207200768e-5, 2.366133620713721e-5],
      linf = [6.156423568670633e-5, 3.474803933150424e-5, 0.00011882557236665703, 0.0001188255723667403, 6.861887285491974e-6, 2.220446049250313e-16, 0.00012195115526943134, 0.00012195115526922318])
  end

end

end # module
