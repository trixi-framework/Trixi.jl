module TestExamplesStructuredMesh1D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "structured_1d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "StructuredMesh1D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [6.0388296447998465e-6],
      linf = [3.217887726258972e-5])
  end

  @trixi_testset "elixir_advection_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
      l2   = [5.641921365468918e-5],
      linf = [0.00021049780975179733])
  end

  @trixi_testset "elixir_advection_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_shockcapturing.jl"),
      l2   = [0.08004076716881656],
      linf = [0.6342577638501385],
      atol = 1.0e-5)
  end

  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [3.67478226e-01, 3.49491179e-01, 8.08910759e-01],
      linf = [1.58971947e+00, 1.59812384e+00, 1.94732969e+00],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [2.2527950196212703e-8, 1.8187357193835156e-8, 7.705669939973104e-8],
      linf = [1.6205433861493646e-7, 1.465427772462391e-7, 5.372255111879554e-7])
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [3.8099996914101204e-6, 1.6745575717106341e-6, 7.732189531480852e-6],
      linf = [1.2971473393186272e-5, 9.270328934274374e-6, 3.092514399671842e-5])
  end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive=true)

end # module
