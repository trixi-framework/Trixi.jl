module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "MHD Multicomponent" begin

  @trixi_testset "elixir_mhdmulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
      l2   = [4.32582228e-02, 4.32415798e-02, 2.58839603e-02, 1.62161967e-01, 1.77570801e-02, 1.77592010e-02, 2.69439008e-02, 3.46643251e-03, 1.20788615e-02, 2.41577231e-02],
      linf = [3.25832463e-01, 3.17478998e-01, 1.87034089e-01, 9.63164821e-01, 8.96912869e-02, 9.07817451e-02, 1.43464850e-01, 4.78963235e-02, 8.48016747e-02, 1.69603349e-01])
  end

  @trixi_testset "elixir_mhdmulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_es.jl"),
      l2   = [4.25355150e-02, 4.25297186e-02, 2.38587767e-02, 1.15514310e-01, 1.64036035e-02, 1.64035840e-02, 2.58227124e-02, 3.21262341e-04, 1.08239236e-02, 2.16478473e-02],
      linf = [2.34122852e-01, 2.34251258e-01, 1.18801302e-01, 5.33830143e-01, 6.21243112e-02, 6.21473584e-02, 9.76738945e-02, 4.73705420e-03, 6.19646775e-02, 1.23929355e-01])
  end

  @trixi_testset "elixir_mhdmulti_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_convergence.jl"),
      l2   = [3.74130370e-04, 3.74130370e-04, 5.17248779e-04, 5.80253989e-04, 4.45245823e-04, 4.45245823e-04, 5.10916997e-04, 3.86209938e-04, 7.52418700e-05, 1.50483740e-04, 3.00967480e-04],
      linf = [1.39240914e-03, 1.39240914e-03, 2.59895755e-03, 1.64731259e-03, 2.02634223e-03, 2.02634223e-03, 2.51710814e-03, 1.16280094e-03, 2.87224446e-04, 5.74448892e-04, 1.14889778e-03])
  end

  @trixi_testset "elixir_mhdmulti_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_rotor.jl"),
      l2   = [3.14385805e-03, 3.14385805e-03, 4.44615350e-03, 4.67667255e-05, 3.14397741e-03, 3.14397741e-03, 4.44613397e-03, 1.99646145e-05, 5.05839422e-06, 1.01167884e-05, 2.02335769e-05],
      linf = [4.54866658e-03, 4.54866658e-03, 6.44493251e-03, 1.72702115e-04, 4.57608106e-03, 4.57608106e-03, 6.41828531e-03, 7.45211429e-05, 1.91820006e-05, 3.83640012e-05, 7.67280025e-05],
      tspan = (0.0, 0.01))
end

end

end # module
