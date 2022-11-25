module TestTree1DFDSBP

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_fdsbp")

@testset "Compressible Euler" begin
  @trixi_testset "elixir_euler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
      l2   = [4.1370344463620254e-6, 4.297052451817826e-6, 9.857382045003056e-6],
      linf = [1.675305070092392e-5, 1.3448113863834266e-5, 3.8185336878271414e-5],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.0003170117861692315, 3.768776739306464e-5, 0.002158213505762533],
      linf = [0.0013022779097824344, 0.0001614330442275269, 0.0067992370102274435],
      tspan = (0.0, 0.1))
  end
end

end # module
