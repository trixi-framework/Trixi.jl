module TestExamples3D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

@testset "3D" begin

# Run basic tests
@testset "Examples 3D" begin
  @testset "taal-confirmed elixir_advection_basic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.00015975754755823664],
      linf = [0.001503873297666436])
  end

  @testset "taal-confirmed elixir_advection_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [9.773858425669403e-6],
      linf = [0.0005853874124926092])
  end


  @testset "taal-confirmed elixir_hyp_diff_llf.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_llf.jl"),
      l2   = [0.0015303292770225546, 0.011314166522881952, 0.011314166522881981, 0.011314166522881947],
      linf = [0.022634590339093097, 0.10150613595329361, 0.10150613595329361, 0.10150613595329361],
      initial_refinement_level=2)
  end

  @testset "elixir_hyp_diff_nonperiodic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_nonperiodic.jl"),
      l2   = [0.00022868340901898148, 0.0007974312252173769, 0.0015035143230655171, 0.0015035143230655694],
      linf = [0.0016405261410663563, 0.0029871222930526976, 0.009410031618266146, 0.009410031618266146])
  end


  @testset "taal-confirmed elixir_euler_source_terms.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.010323099666828388, 0.00972876713766357, 0.00972876713766343, 0.009728767137663324, 0.015080409341036285],
      linf = [0.034894880154510144, 0.03383545920056008, 0.033835459200560525, 0.03383545920054587, 0.06785780622711979])
  end

  @testset "elixir_euler_mortar.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_mortar.jl"),
      l2   = [0.0019011097544691046, 0.0018289464161846331, 0.0018289464161847266, 0.0018289464161847851, 0.0033547668596639966],
      linf = [0.011918626829790169, 0.011808582902362641, 0.01180858290237552, 0.011808582902357312, 0.024648094686513744])
  end


  @testset "taal-confirmed elixir_euler_taylor_green_vortex.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      l2   = [0.0003494971047256544, 0.03133386380969968, 0.031333863809699644, 0.04378595081016185, 0.015796569210801217],
      linf = [0.0013934701399120897, 0.07284947983025436, 0.07284947983025408, 0.12803234075782724, 0.07624639122292365],
  end

  @testset "taal-confirmed elixir_euler_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.025558219399128387, 0.01612806446620796, 0.016128064466207948, 0.016120400619198158, 0.09208276987000782],
      linf = [0.3950327737713353, 0.26324766244272796, 0.2632476624427279, 0.2634129727753079, 1.371321006006725])
  end

  @testset "elixir_euler_density_pulse.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_pulse.jl"),
      l2   = [0.05719652660597408, 0.0571965266059741, 0.05719652660597407, 0.05719652660597409, 0.08579478990896279],
      linf = [0.27375961853433606, 0.27375961853433517, 0.27375961853433384, 0.2737596185343343, 0.4106394278015033],
  end

  @testset "taal-confirmed elixir_euler_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.025101741317688664, 0.01655620530022176, 0.016556205300221737, 0.016549388264402515, 0.09075092792976944],
      linf = [0.43498932208478724, 0.2821813924028202, 0.28218139240282025, 0.2838043627560838, 1.5002293438086647]
  end

  @testset "taal-confirmed elixir_mhd_alfven_wave.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.0038729054515012624, 0.00903693761037057, 0.0041729297273898815, 0.01160504558506348, 0.006241548790045999, 0.009227641613254402, 0.0034580608435846143, 0.011684993365513006, 0.0022068452165023645],
      linf = [0.012628629484152443, 0.03265276295369954, 0.012907838374176334, 0.044746702024108326, 0.02796611265824822, 0.03453054781110626, 0.010261557301859958, 0.044762592434299864, 0.010012319622784436])
  end


  @testset "taal-confirmed elixir_mhd_ec.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.01921453037426997, 0.01924853398980921, 0.01924853398980923, 0.019247118340533328, 0.08310482412935676, 0.010362656540935251, 0.010362656540935237, 0.010364587080559528, 0.00020760700572485828],
      linf = [0.2645851360519166, 0.33611482816103344, 0.33611482816103466, 0.36952265576762666, 1.230825809630423, 0.09818527443798974, 0.09818527443798908, 0.10507242371450054, 0.008456471524217968])
  end


  @testset "taal-check-me elixir_eulergravity_eoc_test.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_eoc_test.jl"),
      l2   = [0.00042767201750631214, 0.0004720121013484361, 0.00047201210134851195, 0.00047201210134847107, 0.0010986046486879376],
      linf = [0.003497353351708421, 0.0037653614087260756, 0.003765361408728074, 0.0037653614087242993, 0.008372792646797134],
      resid_tol = 1.0e-4, tspan = (0.0, 0.2))
  end
end


@testset "Displaying components 3D" begin
  @test_nowarn include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"))

  # test both short and long printing formats
  @test_nowarn show(mesh); println()
  @test_nowarn println(mesh)
  @test_nowarn display(mesh)

  @test_nowarn show(equations); println()
  @test_nowarn println(equations)
  @test_nowarn display(equations)

  @test_nowarn show(solver); println()
  @test_nowarn println(solver)
  @test_nowarn display(solver)

  @test_nowarn show(solver.basis); println()
  @test_nowarn println(solver.basis)
  @test_nowarn display(solver.basis)

  @test_nowarn show(solver.mortar); println()
  @test_nowarn println(solver.mortar)
  @test_nowarn display(solver.mortar)

  @test_nowarn show(semi); println()
  @test_nowarn println(semi)
  @test_nowarn display(semi)

  @test_nowarn show(summary_callback); println()
  @test_nowarn println(summary_callback)
  @test_nowarn display(summary_callback)

  @test_nowarn show(amr_controller); println()
  @test_nowarn println(amr_controller)
  @test_nowarn display(amr_controller)

  @test_nowarn show(amr_callback); println()
  @test_nowarn println(amr_callback)
  @test_nowarn display(amr_callback)

  @test_nowarn show(stepsize_callback); println()
  @test_nowarn println(stepsize_callback)
  @test_nowarn display(stepsize_callback)

  @test_nowarn show(save_solution); println()
  @test_nowarn println(save_solution)
  @test_nowarn display(save_solution)

  @test_nowarn show(analysis_callback); println()
  @test_nowarn println(analysis_callback)
  @test_nowarn display(analysis_callback)

  @test_nowarn show(alive_callback); println()
  @test_nowarn println(alive_callback)
  @test_nowarn display(alive_callback)

  @test_nowarn println(callbacks)
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 3D

end #module
