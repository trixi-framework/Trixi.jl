module TestExamples2DEuler

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Moist Compressible Euler" begin

  @trixi_testset "elixir_moist_euler_source_terms_moist.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_moist_euler_source_terms_moist.jl"),
      l2   = [0.0014110657634349556, 0.8140559401727434, 0.815054860081394, 9.015476455966578,
              2.1382549660881177e-8, 3.000854812562316e-8],
      linf = [0.008475409026544423, 1.9573616670468286, 1.9588026912996557, 54.69222788424668,
              1.2247606173348773e-7 ,1.737143910273667e-7])
  end
end

end # module
