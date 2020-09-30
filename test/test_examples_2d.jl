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
            l2   = [0.12533080510721473],
            linf = [0.9999802982947753])
  end

  @testset "parameters_source_terms.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_source_terms.jl"),
            l2   = [8.517783186497567e-7, 1.2350199409361865e-6, 1.2350199409828616e-6, 4.277884398786315e-6],
            linf = [8.357934254688004e-6, 1.0326389653148027e-5, 1.0326389654924384e-5, 4.4961900057316484e-5])
  end

  @testset "parameters_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_ec.jl"),
            l2   = [0.061733846713578594, 0.05020086119442834, 0.05020836856347214, 0.2259064869636338],
            linf = [0.29894122391731826, 0.30853631977725215, 0.3084722538869674, 1.0652455597305965])
  end

  @testset "parameters_blast_wave_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_blast_wave_shockcapturing.jl"),
            l2   = [0.13575932799459445, 0.11346025131402862, 0.11346028941202581, 0.33371846538168354],
            linf = [1.4662633480487193, 1.3203905049492335, 1.320390504949303, 1.8131376065886553],
            tspan = (0.0, 0.13))
  end

  @testset "parameters_blast_wave_shockcapturing_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_blast_wave_shockcapturing_amr.jl"),
            l2   = [0.6711662708074747, 0.27954874379108824, 0.27954858242154956, 0.7209399244637021],
            linf = [2.8867933412168636, 1.806408513893379, 1.8063306403855512, 3.0345451449653815],
            tspan = (0.0, 1.0))
  end

  @testset "jeans_instability.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "jeans_instability.jl"),
            l2   = [21174.220714494546, 978.743334383856, 5.0322938770733495e-6, 52935.53057317166],
            linf = [29951.893960533664, 1388.4393812604685, 1.2366523531186759e-5, 74879.81397015974],
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
