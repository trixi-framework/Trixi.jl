module TestExamples1DMHD

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "MHD Multicomponent" begin

  @trixi_testset "elixir_mhdmulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
      l2   = [8.16048158e-02, 5.46791194e-02, 5.46791194e-02, 1.54509265e-01, 4.13046273e-17, 5.47637521e-02, 5.47637521e-02, 8.37156486e-03, 1.67431297e-02, 3.34862594e-02],
      linf = [1.81982581e-01, 9.13611439e-02, 9.13611439e-02, 4.23831370e-01, 1.11022302e-16, 9.93731761e-02, 9.93731761e-02, 1.57164285e-02, 3.14328569e-02, 6.28657139e-02])
  end

  @trixi_testset "elixir_mhdmulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_es.jl"),
      l2   = [7.96871667e-02, 5.39815611e-02, 5.39815611e-02, 1.50151690e-01, 4.13046273e-17, 5.36296787e-02, 5.36296787e-02, 8.27911910e-03, 1.65582382e-02, 3.31164764e-02],
      linf = [1.41195919e-01, 7.82015593e-02, 7.82015593e-02, 3.39107175e-01, 1.11022302e-16, 7.00030105e-02, 7.00030105e-02, 1.49425583e-02, 2.98851165e-02, 5.97702330e-02])
  end

  @trixi_testset "elixir_mhdmulti_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_convergence.jl"),
      l2   = [1.84042774e-05, 3.38781451e-04, 3.38781451e-04, 6.70988781e-05, 4.13046273e-17, 3.34549620e-04, 3.34549620e-04, 2.64858640e-05, 5.29717279e-05, 1.05943456e-04],
      linf = [3.94780688e-05, 1.24292799e-03, 1.24292799e-03, 1.78659026e-04, 1.11022302e-16, 1.24361621e-03, 1.24361621e-03, 5.91550778e-05, 1.18310156e-04, 2.36620311e-04])
    end

  @trixi_testset "elixir_mhdmulti_briowu_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_briowu_shock_tube.jl"),
      l2   = [1.93221943e-01, 3.56599349e-01, 0.00000000e+00, 3.63446684e-01, 6.33435897e-16, 3.66051482e-01, 0.00000000e+00, 5.65287156e-02, 1.13057431e-01],
      linf = [4.33927958e-01, 1.09948372e+00, 0.00000000e+00, 1.03943772e+00, 2.99760217e-15, 1.50960600e+00, 0.00000000e+00, 1.96228180e-01, 3.92456360e-01])
    end

end

end # module
