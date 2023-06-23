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
  
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive=true)

end # module
