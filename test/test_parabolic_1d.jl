module TestExamplesParabolic1D

using Test
using Trixi

include("test_trixi.jl")


# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "SemidiscretizationHyperbolicParabolic (1D)" begin

  @trixi_testset "TreeMesh1D: elixir_advection_diffusion.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_advection_diffusion.jl"),
      initial_refinement_level = 4, tspan=(0.0, 0.4), polydeg=3,
      l2 = [8.389498188525518e-06],
      linf = [2.847421658558336e-05]
    )
  end

  @trixi_testset "TreeMesh1D: elixir_navierstokes_convergence.jl" begin
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem", "elixir_navierstokes_convergence.jl"),
      l2 = [0.0001133835907077494, 6.226282245610444e-5, 0.0002820171699999139],
      linf = [0.0006255102377159538, 0.00036195501456059986, 0.0016147729485886941]
    )
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
