module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "MHD Multicomponent" begin

  @trixi_testset "elixir_mhdmulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
      l2   = [4.32946856e-02, 4.32790096e-02, 2.58822718e-02, 1.62024804e-01, 1.77548775e-02, 1.77564324e-02, 2.69452591e-02, 3.45826315e-03, 1.20451485e-02, 2.40902969e-02],
      linf = [3.28937103e-01, 3.31201677e-01, 1.86307657e-01, 1.03382043e+00, 8.77373400e-02, 8.85611301e-02, 1.46918707e-01, 4.84055914e-02, 8.80607765e-02, 1.76121553e-01])
  end

  @trixi_testset "elixir_mhdmulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_es.jl"),
      l2   = [4.25336086e-02, 4.25277388e-02, 2.38503184e-02, 1.15505865e-01, 1.63985643e-02, 1.63986403e-02, 2.58201977e-02, 3.18799432e-04, 1.08086571e-02, 2.16173142e-02],
      linf = [2.34445200e-01, 2.34641311e-01, 1.18747959e-01, 5.33819926e-01, 6.20443752e-02, 6.20664068e-02, 9.73946982e-02, 4.75640645e-03, 6.21579619e-02, 1.24315924e-01])
  end

  @trixi_testset "elixir_mhdmulti_eoc.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_eoc.jl"),
      l2   = [2.59813133e-04, 2.59813133e-04, 4.80423374e-04, 3.28657614e-04, 3.67440987e-04, 3.67440987e-04, 4.99348315e-04, 3.88655564e-04, 5.37357512e-05, 1.07471502e-04, 2.14943005e-04],
      linf = [1.54109844e-03, 1.54109844e-03, 2.39709057e-03, 1.01030991e-03, 1.88632092e-03, 1.88632092e-03, 2.59329609e-03, 1.08934177e-03, 2.13125874e-04, 4.26251747e-04, 8.52503495e-04])
  end

  @trixi_testset "elixir_mhdmulti_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_rotor.jl"),
      l2   = [7.23675501e-01, 7.46495099e-01, 0.00000000e+00, 8.17974304e-01, 1.30738332e-01, 9.21666349e-02, 0.00000000e+00, 1.61636223e-02, 1.97849351e-01, 9.89246753e-02],
      linf = [1.07823615e+01, 1.11367261e+01, 0.00000000e+00, 1.22971727e+01, 2.28094366e+00, 2.93333365e+00, 0.00000000e+00, 4.07011671e-01, 3.44365149e+00, 1.72182574e+00],
      tspan = (0.0, 0.01))
end

end

end # module
