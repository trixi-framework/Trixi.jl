module TestExamples3DHypDiff

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_3d_dgsem")

@testset "Hyperbolic diffusion" begin
  @trixi_testset "elixir_hypdiff_lax_friedrichs.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
      l2   = [0.001530331609036682, 0.011314177033289238, 0.011314177033289402, 0.011314177033289631],
      linf = [0.02263459033909354, 0.10139777904683545, 0.10139777904683545, 0.10139777904683545],
      initial_refinement_level=2)
  end

  @trixi_testset "elixir_hypdiff_lax_friedrichs.jl with surface_flux=flux_godunov)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
      l2   = [0.0015377731806850128, 0.01137685274151801, 0.011376852741518175, 0.011376852741518494],
      linf = [0.022715420630041172, 0.10183745338964201, 0.10183745338964201, 0.1018374533896429],
      initial_refinement_level=2, surface_flux=flux_godunov)
  end

  @trixi_testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
      l2   = [0.00022868320512754316, 0.0007974309948540525, 0.0015035143230654987, 0.0015035143230655293],
      linf = [0.0016405001653623241, 0.0029870057159104594, 0.009410031618285686, 0.009410031618287462])
  end
end

end # module
