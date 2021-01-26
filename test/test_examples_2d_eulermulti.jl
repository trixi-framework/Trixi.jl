module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Compressible Euler Multicomponent" begin
  @testset "elixir_eulermulti_shock_bubble.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble.jl"),
      l2   = [7.38219722e+01, 8.80169627e-01, 5.79496021e+04, 1.83101743e-01, 1.11794879e-02],
      linf = [1.96834444e+02, 7.19013908e+00, 1.58095793e+05, 8.19793567e-01, 8.00831673e-02],
      tspan = (0.0, 0.001))
  end

  @testset "elixir_eulermulti_shock_bubble_3comp.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble_3comp.jl"),
      l2   = [6.69469420e+01, 1.24137635e+00, 5.19736818e+04, 1.67537081e-01, 6.94144768e-03, 7.33198795e-02],
      linf = [2.14068118e+02, 1.82870581e+01, 1.93352191e+05, 1.02911816e+00, 7.00111358e-02, 9.34275213e-01],
      tspan = (0.0, 0.001))
  end

  @testset "elixir_eulermulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
      l2   = [3.48269990e-02, 3.48343748e-02, 9.25283610e-02, 2.10775811e-02, 4.21551623e-02],
      linf = [2.41913824e-01, 2.37051239e-01, 5.64790458e-01, 1.33967766e-01, 2.67935533e-01])
  end

  @testset "elixir_eulermulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
      l2   = [3.45819661e-02, 3.45835287e-02, 9.20021458e-02, 2.09685214e-02, 4.19370428e-02],
      linf = [2.11393171e-01, 2.11289025e-01, 4.89366507e-01, 1.14494265e-01, 2.28988531e-01])
  end

  @testset "elixir_eulermulti_eoc_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_eoc_ec.jl"),
      l2   = [1.15269058e-04, 1.15269058e-04, 1.66578747e-04, 4.61097337e-05, 9.22194673e-05],
      linf = [5.05618608e-04, 5.05618608e-04, 7.87591809e-04, 2.06476960e-04, 4.12953919e-04])
  end

  @testset "elixir_eulermulti_eoc_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_eoc_es.jl"),
      l2   = [1.74779183e-06, 1.74779183e-06, 5.62550396e-06, 6.15682712e-07, 1.23136542e-06],
      linf = [1.51966473e-05, 1.51966473e-05, 5.51065526e-05, 5.58096521e-06, 1.11619304e-05])
  end
end


end # module
