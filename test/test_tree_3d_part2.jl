module TestExamplesTreeMesh3DPart2

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "TreeMesh3D Part 2" begin

# Run basic tests
@testset "Examples 3D" begin
  # Linear scalar advection
  include("test_tree_3d_advection.jl")

  # Hyperbolic diffusion
  include("test_tree_3d_hypdiff.jl")

  # Compressible Euler with self-gravity
  include("test_tree_3d_eulergravity.jl")
end


@trixi_testset "Additional tests in 3D" begin
  @trixi_testset "compressible Euler" begin
    eqn = CompressibleEulerEquations3D(1.4)

    @test isapprox(energy_total([1.0, 2.0, 3.0, 4.0, 20.0], eqn), 20.0)
    @test isapprox(energy_kinetic([1.0, 2.0, 3.0, 4.0, 20], eqn), 14.5)
    @test isapprox(energy_internal([1.0, 2.0, 3.0, 4.0, 20], eqn), 5.5)
  end

  @trixi_testset "hyperbolic diffusion" begin
    @test_nowarn HyperbolicDiffusionEquations3D(nu=1.0)
    eqn = HyperbolicDiffusionEquations3D(nu=1.0)

    @test isapprox(initial_condition_sedov_self_gravity(collect(1:3), 4.5, eqn), zeros(4))
    @test isapprox(boundary_condition_sedov_self_gravity(collect(1:4), 1, 1, collect(11:13), 2.3, flux_central, eqn), [-1.0, -19.739208802178712, 0.0, 0.0])
    @test isapprox(boundary_condition_sedov_self_gravity(collect(1:4), 2, 2, collect(11:13), 4.5, flux_central, eqn), [-1.5, 0.0, -19.739208802178712, 0.0])
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # TreeMesh3D Part 2

end #module
