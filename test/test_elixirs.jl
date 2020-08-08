using LinearAlgebra
using Test
import Trixi

# Start with a clean environment: remove Trixi output directory if it exists
outdir = joinpath(@__DIR__, "out")
isdir(outdir) && rm(outdir, recursive=true)


@testset "Elixirs" begin
  @testset "Test linear structure" begin
    solver, A, b = Trixi.compute_linear_structure("../examples/parameters.toml", initial_refinement_level=2)
    λ = eigvals(Matrix(A))
    @test maximum(real, λ) < length(λ) * eps(real(eltype(λ)))
  end

  @testset "Test linear steady state benchmarks" begin
    Trixi.benchmark_linear_steady_state("../examples/parameters_hyp_diff_nonperiodic.toml",
                                        Trixi.source_terms_poisson_nonperiodic, resid_tol=1.0e-5)
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)
