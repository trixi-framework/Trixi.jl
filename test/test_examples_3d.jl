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


  @testset "elixir_euler_taylor_green_vortex.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
    l2   = [0.0015040212437368603, 0.2789329218326114, 0.27893292183261237, 0.2124964098560632, 0.32470622624447526],
    linf = [0.012164089985932103, 1.4346175439010938, 1.4346175439009117, 1.242662948547226, 2.6937193352995052])
  end

  @testset "elixir_euler_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
    l2   = [0.01948803219944478, 0.013572377993230113, 0.013572377993230083, 0.013559504611364608, 0.07065364959065795],
    linf = [0.19054185048160754, 0.23111606104899296, 0.23111606104899302, 0.20819549633732257, 0.7069189497358219])
  end

  @testset "elixir_euler_density_pulse.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_pulse.jl"),
    l2   = [0.05719653010913493, 0.057196530109134995, 0.057196530109134974, 0.05719653010913503, 0.08579479516370271],
    linf = [0.2741524961597339, 0.27415249615973347, 0.2741524961597339, 0.274152496159735, 0.41122874423960276])
  end

  @testset "elixir_euler_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
    l2   = [0.025279625678119623, 0.016639460251717957, 0.016639460251717947, 0.01663291354143171, 0.09140063001443112],
    linf = [0.43997373040112175, 0.287404037174804, 0.2874040371748037, 0.2892796201004814, 1.5155258175016093])
  end

  @testset "elixir_mhd_alfven_wave.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
    l2   = [0.0038738661034704125, 0.009037180073786493, 0.004170786122172154, 0.011603806159981751, 0.0062502092450648075, 0.009224722367531584, 0.003459461499238398, 0.011683811689607632, 0.0021983399070141674],
    linf = [0.012633279141878506, 0.0326736523735531, 0.01292372050545837, 0.04445719221507506, 0.027990560478335258, 0.03454996296127533, 0.010220142796804721, 0.04499671504008128, 0.009849090317107453])
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
