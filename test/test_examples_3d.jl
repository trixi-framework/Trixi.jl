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
  @testset "elixir_advection_basic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.00015975754755823664],
      linf = [0.001503873297666436])
  end

  @testset "elixir_advection_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
            l2   = [9.773858425669403e-6],
            linf = [0.0005853874124926092])
  end


  @testset "elixir_hyp_diff_llf.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_llf.jl"),
      l2   = [0.0015303316090388799, 0.011314177033289297, 0.011314177033289444, 0.011314177033289696],
      linf = [0.02263459034012283, 0.10139777904690916, 0.10139777904690916, 0.10139777904690828],
      initial_refinement_level=2)
  end

  @testset "elixir_hyp_diff_nonperiodic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_nonperiodic.jl"),
      l2   = [0.00022868340901898148, 0.0007974312252173769, 0.0015035143230655171, 0.0015035143230655694],
      linf = [0.0016405261410663563, 0.0029871222930526976, 0.009410031618266146, 0.009410031618266146])
  end


  @testset "elixir_euler_source_terms.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.010323099666828388, 0.00972876713766357, 0.00972876713766343, 0.009728767137663324, 0.015080409341036285],
      linf = [0.034894880154510144, 0.03383545920056008, 0.033835459200560525, 0.03383545920054587, 0.06785780622711979])
  end

  @testset "elixir_euler_mortar.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_mortar.jl"),
      l2   = [0.0019011097544691046, 0.0018289464161846331, 0.0018289464161847266, 0.0018289464161847851, 0.0033547668596639966],
      linf = [0.011918626829790169, 0.011808582902362641, 0.01180858290237552, 0.011808582902357312, 0.024648094686513744])
  end

  @testset "elixir_euler_density_pulse.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_pulse.jl"),
    l2   = [0.07822491609408178, 0.09094495898139875, 0.09094495898139938, 0.09094495898139882, 0.2498601494336289],
    linf = [0.27610733371342255, 0.3068393585577063, 0.30683935855770783, 0.30683935855770783, 0.6384108081914857])
  end

  @testset "elixir_euler_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
    l2   = [0.025279625678119623, 0.016639460251717957, 0.016639460251717947, 0.01663291354143171, 0.09140063001443112],
    linf = [0.43997373040112175, 0.287404037174804, 0.2874040371748037, 0.2892796201004814, 1.5155258175016093])
  end

  @testset "elixir_mhd_ec.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.017285648572570363, 0.01777055834391421, 0.01777055834391415, 0.017772303802227787, 0.07402246754850351, 0.010363311581708652, 0.010363311581708655, 0.010365244788128367, 0.00020795117986261875],
      linf = [0.26483877701302616, 0.3347840483592971, 0.3347840483592973, 0.3698107272043008, 1.2339463134928033, 0.09858876654056647, 0.09858876654056714, 0.10426402075606456, 0.008001763586594345])
  end


  @testset "elixir_eulergravity_eoc_test.jl" begin
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
