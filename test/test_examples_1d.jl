module TestExamples1D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

@testset "1D" begin

# Run basic tests
@testset "Examples 1D" begin
  @testset "elixir_advection_basic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [6.0388296447998465e-6],
      linf = [3.217887726258972e-5])
  end

  @testset "elixir_advection_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
    l2   = [0.3540209654959832],
    linf = [0.9999905446337742])
  end


  @testset "elixir_euler_source_terms.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
    l2   = [2.3017487546631118e-8, 1.9137514928527178e-8, 7.732828544277078e-8],
    linf = [1.6282751857943367e-7, 1.426988238684146e-7, 5.298297782729833e-7])
  end

  @testset "elixir_euler_nonperiodic.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic.jl"),
    l2   = [3.80950031272884e-6, 1.671745083458876e-6, 7.730956863549413e-6],
    linf = [1.2966215741316844e-5, 9.2635164730126e-6, 3.090770562241829e-5])
  end

  @testset "elixir_euler_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
    l2   = [0.1188410508446165, 0.15463666913848456, 0.4444355816866067],
    linf = [0.23934128951004474, 0.27246473813214184, 0.8697154266487717])
  end

  @testset "elixir_euler_blast_wave_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_shockcapturing.jl"),
      l2   = [0.21218593029900773, 0.2769530413665288, 0.5518482111667276],
      linf = [1.505221631144809, 1.4864840122024228, 2.0501644413816162],
      tspan = (0.0, 0.13))
  end
end


@testset "Displaying components 1D" begin
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

  @test_nowarn show(amr_indicator); println()
  @test_nowarn println(amr_indicator)
  @test_nowarn display(amr_indicator)

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

end # 1D

end #module
