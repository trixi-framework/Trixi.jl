module TestExamples2D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

# Run basic tests
@testset "Examples 2D" begin
  @testset "parameters.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "parameters.jl"),
            l2   = [9.144681765639205e-6],
            linf = [6.437440532547356e-5])
  end
  @testset "parameters_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_ec.jl"),
            l2   = [0.06169846095311385, 0.05016515041472451, 0.05017264946347607, 0.22577667054257733],
            linf = [0.2954432920699207, 0.30754595417690045, 0.3074869003416839, 1.053744736882769])
  end
  @testset "parameters_mortar.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "parameters_mortar.jl"),
            l2   = [0.022356422238096973],
            linf = [0.5043638249003257])
  end
end


@testset "Displaying components 2D" begin
  @test_nowarn include(joinpath(EXAMPLES_DIR, "parameters.jl"))
  display(mesh)
  display(equations)
  display(solver)
  display(semi)
  display(stepsize_callback)
  display(analysis_callback)
  display(save_solution)
  display(alive_callback)
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
