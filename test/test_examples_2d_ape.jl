module TestExamples2DAPE

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Acoustic Perturbation" begin
  @testset "elixir_ape_convergence_test.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_convergence_test.jl"),
      l2   = [0.0018873176537420558, 0.00228615992541537, 0.003125083031985467],
      linf = [0.007481011083990019, 0.009788636350308355, 0.02522255492947867])
  end

  @testset "elixir_ape_gauss.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_gauss.jl"),
      l2 = [0.08005276517890283, 0.08005276517890268, 0.4187202920734123],
      linf = [0.17261097190220992, 0.17261097190220973, 1.13601894068238])
  end

  @testset "elixir_ape_gaussian_source.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_gaussian_source.jl"),
      l2 = [0.004296394903650806, 0.004241280404758938, 0.006269684906035964],
      linf = [0.03970270697049378, 0.04151096349298151, 0.0640019829058819])
  end
end

end # module