module TestExamples2D

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
end


# @testset "Displaying components 3D" begin
#   @test_nowarn include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"))

#   # test both short and long printing formats
#   @test_nowarn show(mesh); println()
#   @test_nowarn println(mesh)
#   @test_nowarn display(mesh)

#   @test_nowarn show(equations); println()
#   @test_nowarn println(equations)
#   @test_nowarn display(equations)

#   @test_nowarn show(solver); println()
#   @test_nowarn println(solver)
#   @test_nowarn display(solver)

#   @test_nowarn show(solver.basis); println()
#   @test_nowarn println(solver.basis)
#   @test_nowarn display(solver.basis)

#   @test_nowarn show(solver.mortar); println()
#   @test_nowarn println(solver.mortar)
#   @test_nowarn display(solver.mortar)

#   @test_nowarn show(semi); println()
#   @test_nowarn println(semi)
#   @test_nowarn display(semi)

#   @test_nowarn show(summary_callback); println()
#   @test_nowarn println(summary_callback)
#   @test_nowarn display(summary_callback)

#   @test_nowarn show(amr_controller); println()
#   @test_nowarn println(amr_controller)
#   @test_nowarn display(amr_controller)

#   @test_nowarn show(amr_callback); println()
#   @test_nowarn println(amr_callback)
#   @test_nowarn display(amr_callback)

#   @test_nowarn show(stepsize_callback); println()
#   @test_nowarn println(stepsize_callback)
#   @test_nowarn display(stepsize_callback)

#   @test_nowarn show(save_solution); println()
#   @test_nowarn println(save_solution)
#   @test_nowarn display(save_solution)

#   @test_nowarn show(analysis_callback); println()
#   @test_nowarn println(analysis_callback)
#   @test_nowarn display(analysis_callback)

#   @test_nowarn show(alive_callback); println()
#   @test_nowarn println(alive_callback)
#   @test_nowarn display(alive_callback)

#   @test_nowarn println(callbacks)
# end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 3D

end #module
