module TestExamples3dP4est

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

@testset "P4estMesh" begin
  @trixi_testset "elixir_advection_basic_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_p4est.jl"),
      l2   = [0.00013446460962856976],
      linf = [0.0012577781391462928])
  end

  @trixi_testset "elixir_advection_p4est_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_p4est_unstructured_curved.jl"),
      l2   = [0.0006244885699399409],
      linf = [0.04076651402041587])
  end

  @trixi_testset "elixir_advection_p4est_non_conforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_p4est_non_conforming.jl"),
      l2   = [0.0024774648310858928],
      linf = [0.021727876954353964])
  end

  @trixi_testset "elixir_advection_amr_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_p4est.jl"),
      l2   = [9.773852895157622e-6],
      linf = [0.0005853874124926162])
  end

  @trixi_testset "elixir_advection_amr_p4est_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_p4est_unstructured_curved.jl"),
      l2   = [0.00014665036779554962],
      linf = [0.00845936405684372])
  end

  @trixi_testset "elixir_advection_restart_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_p4est.jl"),
      l2   = [0.0281388160824776],
      linf = [0.08740635193023694])
  end

  @trixi_testset "elixir_euler_nonperiodic_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic_p4est.jl"),
      l2   = [0.0018268326813744103, 0.001745601029521995, 0.001745601029521962, 0.0017456010295218891, 0.003239834454817457],
      linf = [0.014660503198892005, 0.01506958815284798, 0.01506958815283821, 0.015069588152864632, 0.02700205515651044])
  end

  @trixi_testset "elixir_euler_free_stream_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_p4est.jl"),
      l2   = [8.116891021528916e-15, 3.5522708607626286e-14, 4.143281237333047e-14, 6.289087484242244e-14, 6.4284229587973e-14],
      linf = [5.220268661787486e-13, 1.620439893379455e-12, 2.3271384819167906e-12, 2.466915560717098e-12, 5.472955422192172e-12])
  end
end

end # module
