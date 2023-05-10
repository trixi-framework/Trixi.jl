module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "MHD Multi-ion" begin

  @trixi_testset "elixir_mhdmultiion_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
      l2   = [1.56133690e-02,   1.56211211e-02,   2.44289260e-02,   1.17053210e-02,   1.35748661e-01,
              1.35779534e-01,   1.34646112e-01,   1.34813656e-01,   1.93724876e-02,   2.70357315e-01,
              2.70356924e-01,   2.69252524e-01,   1.86315505e-01],
      linf = [1.06156769e-01,   1.15019769e-01,   1.32816030e-01,   7.65402322e-02,   2.45518940e-01,
              2.46123607e-01,   1.82733442e-01,   4.24743430e-01,   1.27620999e-01,   4.58874938e-01,
              4.65364246e-01,   3.56983044e-01,   3.94035665e-01])
  end

  @trixi_testset "elixir_mhdmultiion_rotor.jl tspan = (0., 0.001)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_rotor.jl"),
      l2   = [8.74074695e-03,   1.52216843e-02,   5.46908589e-06,   4.48801984e-02,   8.64335927e-02,
              8.16206313e-02,   2.79217739e-03,   1.30650336e-01,   4.48808155e-02,   8.64347180e-02,
              8.16273475e-02,   5.58430809e-03,   1.30652670e-01],
      linf = [1.22933218e-01,   2.25300270e-01,   2.89189052e-05,   1.03135219e+00,   3.57199056e+00,
              3.36455287e+00,   1.44792528e-02,   4.94065455e+00,   1.03150012e+00,   3.57211417e+00,   3.36511058e+00,   2.88397185e-02,   4.94152909e+00],
              tspan = (0., 0.001))
  end

end

end # module
