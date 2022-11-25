module TestTree3DFDSBP

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_3d_fdsbp")

@testset "Compressible Euler" begin
  @trixi_testset "elixir_euler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
      l2   = [2.247522803543667e-5, 2.2499169224681058e-5, 2.24991692246826e-5, 2.2499169224684707e-5, 5.814121361417382e-5],
      linf = [9.579357410749445e-5, 9.544871933409027e-5, 9.54487193367548e-5, 9.544871933453436e-5, 0.0004192294529472562],
      tspan = (0.0, 0.2))
  end

  @trixi_testset "elixir_euler_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      l2   = [7.279180604891456e-5, 0.006268185712551974, 0.006268185712552052, 0.008837033095318898, 0.002701809303715869],
      linf = [0.0002759971955125229, 0.013183586257262458, 0.013183586256375723, 0.024949217432277716, 0.010428855388113334],
      tspan = (0.0, 0.1))
  end
end

end # module
