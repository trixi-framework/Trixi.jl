module TestElixirs

using LinearAlgebra
using Test
import Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples")


@testset "Elixirs" begin
  @testset "Test linear structure (2D)" begin
    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "2d", "parameters.toml"),
                                          initial_refinement_level=2)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < length(λ) * eps(real(eltype(λ)))

    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "2d", "parameters_hyp_diff_llf.toml"),
                                          Trixi.source_terms_harmonic,
                                          initial_refinement_level=2)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < 10*length(λ) * eps(real(eltype(λ)))
  end

  @testset "Test linear structure (3D)" begin
    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "3d", "parameters.toml"),
                                          N=2, initial_refinement_level=1)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < length(λ) * eps(real(eltype(λ)))

    A, b = Trixi.compute_linear_structure(joinpath(EXAMPLES_DIR, "3d", "parameters_hyp_diff_llf.toml"),
                                          Trixi.source_terms_harmonic,
                                          N=2, initial_refinement_level=1)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < 10*length(λ) * eps(real(eltype(λ)))
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
