module TestExamplesP4estMesh2D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "p4est_2d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "P4estMesh2D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [8.311947673061856e-6], 
      linf = [6.627000273229378e-5])
  end

  @trixi_testset "elixir_advection_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
      l2   = [2.7905480848832338e-5], 
      linf = [0.00022847020768290704])
  end

  @trixi_testset "elixir_advection_nonconforming_unstructured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming_unstructured.jl"),
      l2   = [0.0026958321660563362], 
      linf = [0.04122573088346193])
  end

  @trixi_testset "elixir_advection_amr_solution_independent.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_solution_independent.jl"),
      # Expected errors are exactly the same as with StructuredMesh!
      l2   = [4.949660644033807e-5], 
      linf = [0.0004867846262313763])
  end

  @trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_unstructured_flag.jl"),
      l2   = [0.0012766060609964525],
      linf = [0.01750280631586159])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [4.219208035626337e-6], 
      linf = [3.4384344042126536e-5])
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [2.2594405120861626e-6, 2.3188881560096627e-6, 2.318888156064081e-6, 6.332786324236605e-6],
      linf = [1.4987382633613322e-5, 1.9182011925522602e-5, 1.9182011924634423e-5, 6.0526717124531615e-5])
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic_unstructured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic_unstructured.jl"),
      l2   = [0.005238881525717585, 0.0043246899191607775, 0.0043246899191606925, 0.00986166157942196],
      linf = [0.052183952487556695, 0.05393708345945791, 0.05393708345946191, 0.09119199890635965])
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [2.063350241405049e-15, 1.8571016296925367e-14, 3.1769447886391905e-14, 1.4104095258528071e-14],
      linf = [1.9539925233402755e-14, 2e-12, 4.8e-12, 4e-12],
      atol = 2.0e-12, # required to make CI tests pass on macOS
    )
  end

  @trixi_testset "elixir_eulergravity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
      l2   = [0.00024871265138964204, 0.0003370077102132591, 0.0003370077102131964, 0.0007231525513793697],
      linf = [0.0015813032944647087, 0.0020494288423820173, 0.0020494288423824614, 0.004793821195083758],
      tspan = (0.0, 0.1))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # module
