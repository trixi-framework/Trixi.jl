module TestExamples2DAPE

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Acoustic Perturbation" begin
  @trixi_testset "elixir_ape_convergence_test.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_convergence_test.jl"),
      l2   = [0.0018873176537420558, 0.00228615992541537, 0.003125083031985467, 0.0, 0.0, 0.0, 0.0],
      linf = [0.007481011083990019, 0.009788636350308355, 0.02522255492947867, 0.0, 0.0, 0.0, 0.0])
  end

  @trixi_testset "elixir_ape_gauss.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_gauss.jl"),
      l2 = [0.08005276517890283, 0.08005276517890268, 0.4187202920734123, 0.0, 0.0, 0.0, 0.0],
      linf = [0.17261097190220992, 0.17261097190220973, 1.13601894068238, 0.0, 0.0, 0.0, 0.0])
  end

  @trixi_testset "elixir_ape_gaussian_source.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_gaussian_source.jl"),
      l2 = [0.004296394903650806, 0.004241280404758938, 0.006269684906035964, 0.0, 0.0, 0.0, 0.0],
      linf = [0.03970270697049378, 0.04151096349298151, 0.0640019829058819, 0.0, 0.0, 0.0, 0.0])
  end

  @trixi_testset "elixir_ape_gauss_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_gauss_wall.jl"),
    l2 = [0.019419398248465843, 0.019510701017551826, 0.04818246051887614,
          7.382060834820337e-17, 0.0, 1.4764121669640674e-16, 1.4764121669640674e-16],
    linf = [0.18193631937316496, 0.1877464607867628, 1.0355388011792845,
            2.220446049250313e-16, 0.0, 4.440892098500626e-16, 4.440892098500626e-16])
  end

  @trixi_testset "elixir_ape_monopole.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_ape_monopole.jl"),
    l2 = [0.006816790293009947, 0.0065068948357351625, 0.008724512056168938,
          0.0009894398191644543, 0.0, 7.144325530679576e-17, 7.144325530679576e-17],
    linf = [1.000633375007386, 0.5599788929862504, 0.5738432957070382,
            0.015590137026938428, 0.0, 2.220446049250313e-16, 2.220446049250313e-16])
  end
end

end # module