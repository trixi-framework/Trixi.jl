module TestExamplesT8codeMesh3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "t8code_3d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)
mkdir(outdir)

@testset "T8codeMesh3D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [0.00016263963870641478],
      linf = [0.0014537194925779984])
  end

  @trixi_testset "elixir_advection_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_curved.jl"),
      l2   = [0.0004750004258546538],
      linf = [0.026527551737137167])
  end

  @trixi_testset "elixir_advection_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
      l2   = [0.00253595715323843],
      linf = [0.016486952252155795])

    # Ensure that we do not have excessive memory allocations 
    # (e.g., from type instabilities)
    let
      t = sol.t[end]
      u_ode = sol.u[end]
      du_ode = similar(u_ode)
      @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
  end

  @trixi_testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [1.1302812803902801e-5],
      linf = [0.0007889950196294793],
      coverage_override = (maxiters=6, initial_refinement_level=1, base_level=1, med_level=2, max_level=3))
  end

  @trixi_testset "elixir_advection_amr_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_unstructured_curved.jl"),
      l2   = [1.6236411810065552e-5],
      linf = [0.0010554006923731395],
      tspan = (0.0, 1.0),
      coverage_override = (maxiters=6, initial_refinement_level=0, base_level=0, med_level=1, max_level=2))
  end

  # @trixi_testset "elixir_advection_cubed_sphere.jl" begin
  #   @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_cubed_sphere.jl"),
  #     l2   = [0.002006918015656413],
  #     linf = [0.027655117058380085])
  # end

end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive=true)

end # module
