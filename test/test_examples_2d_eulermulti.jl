module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Compressible Euler Multicomponent" begin
  @testset "elixir_eulermulti_shock_bubble.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble.jl"),
      l2   = [6.03088526e-02, 2.88034228e-03, 2.41548900e+01, 7.83393342e-02, 1.91342787e+04],
      linf = [7.37128632e-01, 1.05064677e-01, 1.99311122e+02, 2.01502716e+00, 1.57477455e+05],
      tspan = (0.0, 0.0001))
  end

  @testset "elixir_eulermulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
      l2   = [1.23457293e-02, 4.93829171e-02, 5.01948074e-02, 5.02023248e-02, 2.25886833e-01],
      linf = [5.96271450e-02, 2.38508580e-01, 3.06937711e-01, 3.06807092e-01, 1.06295287e+00])
  end

  @testset "elixir_eulermulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
      l2   = [1.22783570e-02, 4.91134280e-02, 4.97009915e-02, 4.97029892e-02, 2.24364821e-01],
      linf = [6.33058769e-02, 2.53223508e-01, 2.52097945e-01, 2.48924663e-01, 9.32036428e-01])
  end

  @testset "elixir_eulermulti_eoc_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_eoc_ec.jl"),
      l2   = [3.41443064e-06, 1.36577226e-05, 1.44713621e-05, 1.44713621e-05, 2.28669565e-05],
      linf = [1.81596016e-05, 7.26384063e-05, 7.36760417e-05, 7.36760417e-05, 1.21009804e-04])
  end

  @testset "elixir_eulermulti_eoc_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_eoc_es.jl"),
      l2   = [5.54801268e-07, 2.21920507e-06, 1.99274552e-06, 1.99274552e-06, 6.05125237e-06],
      linf = [4.60991759e-06, 1.84396703e-05, 1.83105421e-05, 1.83105421e-05, 5.41312537e-05])
  end
end


end # module
