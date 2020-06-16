using LinearAlgebra
using Test
import Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = joinpath(@__DIR__, "out")
isdir(outdir) && rm(outdir, recursive=true)


@testset "Elixirs" begin
  @testset "Test linear structure" begin
    A, b = Trixi.compute_linear_structure("../examples/parameters.toml", initial_refinement_level=2)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < length(λ) * eps(real(eltype(λ)))
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)
