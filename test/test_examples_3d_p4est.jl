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
      l2   = [7.839921025049303e-15, 4.3320953097202227e-14, 4.61318381170666e-14, 5.466388734250446e-14, 9.381101254280883e-14],
      linf = [1.0269562977782698e-12, 4.9040771443742415e-12, 6.0748628349927e-12, 7.261524714863299e-12, 1.1615597372838238e-11],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_euler_free_stream_p4est_extruded.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_p4est_extruded.jl"),
      l2   = [1.6667578354338437e-14, 8.351207753024527e-15, 1.805859250264852e-14, 1.90966219766217e-14, 1.90966219766217e-14],
      linf = [1.0413891970983968e-13, 4.988787161153141e-13, 2.0694557179012918e-13, 6.221689829999377e-13, 1.0587086762825493e-12])
  end

  @trixi_testset "elixir_euler_ec_p4est.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec_p4est.jl"),
      l2   = [0.009411352784942132, 0.007257089010979736, 0.007478550902224631, 0.007472675228591207, 0.023851056842379016],
      linf = [0.19307851386933228, 0.27472353046951714, 0.3039307725248127, 0.3063060931738112, 0.5144856763103725])
  end
end

end # module
