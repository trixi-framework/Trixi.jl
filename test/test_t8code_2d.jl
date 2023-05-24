module TestExamplesT8codeMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "t8code_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)
mkdir(outdir)

@testset "T8codeMesh2D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [8.311947673061856e-6],
      linf = [6.627000273229378e-5])
  end

  @trixi_testset "elixir_advection_nonconforming_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming_flag.jl"),
      l2   = [3.198940059144588e-5],
      linf = [0.00030636069494005547])

    # Ensure that we do not have excessive memory allocations 
    # (e.g., from type instabilities)
    let
      t = sol.t[end]
      u_ode = sol.u[end]
      du_ode = similar(u_ode)
      @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
  end

  @trixi_testset "elixir_advection_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_flag.jl"),
      l2   = [0.0005379687442422346],
      linf = [0.007438525029884735])
  end

  @trixi_testset "elixir_advection_amr_solution_independent.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_solution_independent.jl"),
      # Expected errors are exactly the same as with StructuredMesh!
      l2   = [4.949660644033807e-5],
      linf = [0.0004867846262313763],
      coverage_override = (maxiters=6,))
  end

end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive=true)

end # module
