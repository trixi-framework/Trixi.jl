module TestExamples3DPart2

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "3D-Part2" begin

# Run basic tests
@testset "Examples 3D" begin
  # MHD
  include("test_examples_3d_mhd.jl")

  # Lattice-Boltzmann
  include("test_examples_3d_lbm.jl")

  # Curved mesh
  include("test_examples_3d_curved.jl")
end


@testset "Additional tests in 3D" begin
  @testset "ideal GLM MHD" begin
    eqn = IdealGlmMhdEquations3D(1.4)
    u = [1.0, 2.0, 3.0, 4.0, 20.0, 0.1, 0.2, 0.3, 1.5]

    @test isapprox(density(u, eqn), 1.0)
    @test isapprox(pressure(u, eqn), 1.7219999999999995)
    @test isapprox(density_pressure(u, eqn), 1.7219999999999995)

    @test isapprox(Trixi.entropy_thermodynamic(u, eqn), 0.5434864060055388)
    @test isapprox(Trixi.entropy_math(u, eqn), -1.3587160150138473)
    @test isapprox(Trixi.entropy(u, eqn), -1.3587160150138473)

    @test isapprox(energy_total(u, eqn), 20.0)
    @test isapprox(energy_kinetic(u, eqn), 14.5)
    @test isapprox(energy_magnetic(u, eqn), 0.07)
    @test isapprox(energy_internal(u, eqn), 4.305)

    @test isapprox(cross_helicity(u, eqn), 2.0)
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 3D-Part2

end #module
