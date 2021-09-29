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
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @trixi_testset "elixir_advection_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
      l2   = [2.58174252995893e-5],
      linf = [2.65115939055204e-4])
  end

  @trixi_testset "elixir_advection_nonconforming_unstructured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming_unstructured.jl"),
      l2   = [3.038384623519386e-3],
      linf = [6.324792487776842e-2])
  end

  @trixi_testset "elixir_advection_amr_solution_independent.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_solution_independent.jl"),
      l2   = [7.945073295691892e-5],
      linf = [0.0007454287896710293])
  end

  @trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_unstructured_flag.jl"),
      l2   = [1.29561551e-03],
      linf = [3.92061383e-02])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [6.398955192910044e-6],
      linf = [3.474337336717426e-5])
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

  @trixi_testset "elixir_euler_shockcapturing_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
      l2   = [9.53984675e-02, 1.05633455e-01, 1.05636158e-01, 3.50747237e-01],
      linf = [2.94357464e-01, 4.07893014e-01, 3.97334516e-01, 1.08142520e+00],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [3.76149952e-01, 2.46970327e-01, 2.46970327e-01, 1.28889042e+00],
      linf = [1.22139001e+00, 1.17742626e+00, 1.17742626e+00, 6.20638482e+00],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_euler_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
      l2   = [9.77167251e-02, 7.85814733e-02, 7.85814010e-02, 2.08656399e-01],
      linf = [2.07569959e+00, 1.78963061e+00, 1.78963061e+00, 2.18728474e+00],
      tspan = (0.0, 0.05))
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
