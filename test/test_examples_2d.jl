module TestExamples2D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "2D" begin

# Run basic tests
@testset "Examples 2D" begin
  @testset "parameters.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "parameters.jl"),
            l2   = [9.144681765639205e-6],
            linf = [6.437440532547356e-5])
  end

  @testset "parameters_mortar.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_mortar.jl"),
            l2   = [0.022356422238096973],
            linf = [0.5043638249003257])
  end

  @testset "parameters_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_amr.jl"),
            l2   = [0.12518146315141132],
            linf = [0.9976001873456246])
  end

  @testset "parameters_source_terms.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_source_terms.jl"),
            l2   = [8.120578592425209e-6, 1.0733370429867915e-5, 1.073337042980614e-5, 3.753181552467947e-5],
            linf = [1.875169907283869e-5, 2.487814009000111e-5, 2.48781400904452e-5, 9.571137504771343e-5])
  end

  @testset "parameters_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_ec.jl"),
            l2   = [0.06169846095311385, 0.05016515041472451, 0.05017264946347607, 0.22577667054257733],
            linf = [0.2954432920699207, 0.30754595417690045, 0.3074869003416839, 1.053744736882769])
  end

  @testset "parameters_blast_wave_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_blast_wave_shockcapturing.jl"),
            l2   = [0.13575932799459445, 0.11346025131402862, 0.11346028941202581, 0.33371846538168354],
            linf = [1.4662633480487193, 1.3203905049492335, 1.320390504949303, 1.8131376065886553],
            tspan = (0.0, 0.13))
  end

  @testset "jeans_instability.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "jeans_instability.jl"),
            l2   = [21170.799181833314, 984.2603425617935, 3.818652342299141e-6, 52926.97673548989],
            linf = [29947.077284164727, 1396.2210559760179, 1.1783469325080385e-5, 74867.78039361164],
            tspan = (0.0, 0.6))
  end
end


@testset "Displaying components 2D" begin
  @test_nowarn include(joinpath(EXAMPLES_DIR, "parameters.jl"))

  # test both short and long printing formats
  @test_nowarn println(mesh)
  @test_nowarn display(mesh)

  @test_nowarn println(equations)
  @test_nowarn display(equations)

  @test_nowarn println(solver)
  @test_nowarn display(solver)

  @test_nowarn println(semi)
  @test_nowarn display(semi)

  @test_nowarn println(stepsize_callback)
  @test_nowarn display(stepsize_callback)

  @test_nowarn println(analysis_callback)
  @test_nowarn display(analysis_callback)

  @test_nowarn println(save_solution)
  @test_nowarn display(save_solution)

  @test_nowarn println(alive_callback)
  @test_nowarn display(alive_callback)
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D

end #module
