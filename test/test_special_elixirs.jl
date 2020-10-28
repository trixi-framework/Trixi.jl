module TestElixirs

using LinearAlgebra
using Test
using Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples")


@testset "Special elixirs" begin
  @testset "Test Jacobian of DG (2D)" begin
    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_advection_basic.jl"),
                  tspan=(0.0, 0.0), initial_refinement_level=2)
    J = jacobian_fd(semi)
    λ = eigvals(J)
    @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))


    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "2d", "elixir_euler_density_wave.jl"),
                  tspan=(0.0, 0.0), initial_refinement_level=2)
    J = jacobian_fd(semi)
    λ = eigvals(J)
    @test maximum(real, λ) < 0.007
  end

  @testset "Test Jacobian of DG (3D)" begin
    trixi_include(@__MODULE__, joinpath(EXAMPLES_DIR, "3d", "elixir_advection_basic.jl"),
                  tspan=(0.0, 0.0), initial_refinement_level=1)
    J = jacobian_fd(semi)
    λ = eigvals(J)
    @test maximum(real, λ) < 10 * sqrt(eps(real(semi)))
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
