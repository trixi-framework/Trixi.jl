module TestExamplesDGMulti1D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_1d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "DGMulti 1D" begin
  @trixi_testset "elixir_euler_flux_diff.jl " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"),
      cells_per_dimension = (16,),
      l2 = [7.853842541289665e-7, 9.609905503440606e-7, 2.832322219966481e-6],
      linf = [1.5003758788711963e-6, 1.802998748523521e-6, 4.83599270806323e-6]
    )
  end

  @trixi_testset "elixir_euler_flux_diff.jl (convergence)" begin
    mean_convergence = convergence_test(@__MODULE__, joinpath(EXAMPLES_DIR, "elixir_euler_flux_diff.jl"), 3)
    @test isapprox(mean_convergence[:l2], [4.1558759698638434, 3.977911306037128, 4.041421206468769], rtol=0.05)
  end

end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
