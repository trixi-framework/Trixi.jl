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
  @testset "Test linear structure (2D)" begin
    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "2d", "parameters_advection_basic.toml"),
                                          initial_refinement_level=2)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < length(λ) * eps(real(eltype(λ)))

    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "2d", "parameters_hypdiff_lax_friedrichs.toml"),
                                          Trixi.source_terms_harmonic,
                                          initial_refinement_level=2)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < 10*length(λ) * eps(real(eltype(λ)))
  end

  @testset "Test Jacobian of DG (2D)" begin
    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "2d", "parameters_advection_basic.toml"),
                                          initial_refinement_level=2)
    J = Trixi.compute_jacobian_dg(joinpath(EXAMPLES_DIR, "2d", "parameters_advection_basic.toml"),
                                          initial_refinement_level=2)
    @test isapprox(Matrix(A), J)

    J = Trixi.compute_jacobian_dg(joinpath(EXAMPLES_DIR, "2d", "parameters_euler_density_wave.toml"))
    λ = eigvals(J)
    @test maximum(real, λ) < 0.007
  end

  @testset "Test linear structure (3D)" begin
    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "3d", "parameters_advection_basic.toml"),
                                          polydeg=2, initial_refinement_level=1)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < length(λ) * eps(real(eltype(λ)))

    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "3d", "parameters_hypdiff_lax_friedrichs.toml"),
                                          Trixi.source_terms_harmonic,
                                          polydeg=2, initial_refinement_level=1)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < 10*length(λ) * eps(real(eltype(λ)))
  end

  @testset "Test Jacobian of DG (3D)" begin
    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "3d", "parameters_advection_basic.toml"),
                                          polydeg=2, initial_refinement_level=1)
    J = Trixi.compute_jacobian_dg(joinpath(EXAMPLES_DIR, "3d", "parameters_advection_basic.toml"),
                                          polydeg=2, initial_refinement_level=1)
    @test isapprox(Matrix(A), J)
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
