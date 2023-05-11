module TestExamples1DMHD

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

@testset "MHD Multi-ion" begin

  @trixi_testset "elixir_mhdmultiion_ec_onespecies.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec_onespecies.jl"),
      l2   = [4.13046273e-17,   5.47627735e-02,   5.47627735e-02,   5.85364902e-02,   8.15735949e-02,
              5.46480229e-02,   5.46480229e-02,   1.54430906e-01],
      linf = [1.11022302e-16,   9.62277600e-02,   9.62277600e-02,   1.07398441e-01,   1.85851486e-01,
              9.41669606e-02,   9.41669606e-02,   4.10966135e-01])
  end

  @trixi_testset "elixir_mhdmultiion_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
      l2   = [4.13046273e-17,   4.41300832e-02,   4.08698259e-02,   1.45921842e-02,   1.46195334e-01,
              1.46189460e-01,   1.47069647e-01,   1.15948953e-01,   4.17156345e-02,   2.95429888e-01,   
              2.91864340e-01,   2.90281705e-01,   1.91712252e-01],
      linf = [1.11022302e-16,   8.23475323e-02,   8.20044181e-02,   5.26482770e-02,   2.36978475e-01,
              1.78890885e-01,   1.83844973e-01,   3.69223717e-01,   9.49715344e-02,   4.04059325e-01,
              3.53727376e-01,   3.43908646e-01,   3.72557303e-01])
  end

  @trixi_testset "elixir_mhdmultiion_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_es.jl"),
      l2   = [4.13046273e-17,   4.34692117e-02,   4.01800240e-02,   1.39798040e-02,   1.45588748e-01,
              1.46145114e-01,   1.47018561e-01,   1.07728669e-01,   4.13841438e-02,   2.95261011e-01,
              2.91827041e-01,   2.90260310e-01,   1.90243105e-01],
      linf = [1.11022302e-16,   7.89266630e-02,   7.79256051e-02,   4.76391824e-02,   2.07007992e-01,
              1.79314301e-01,   1.84325683e-01,   3.47578503e-01,   9.30059101e-02,   3.81670634e-01,
              3.53221946e-01,   3.43511206e-01,   3.75916013e-01])
  end

  @trixi_testset "elixir_mhdmultiion_es_shock_capturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_es_shock_capturing.jl"),
      l2   = [4.13046273e-17,   4.34327035e-02,   4.01429579e-02,   1.39648331e-02,   1.45589699e-01,
              1.46145036e-01,   1.47013130e-01,   1.07647870e-01,   4.13842626e-02,   2.95252636e-01,
              2.91824474e-01,   2.90263048e-01,   1.90199794e-01],
      linf = [1.11022302e-16,   7.86144728e-02,   7.75970804e-02,   4.75320603e-02,   2.07019087e-01,
              1.79245486e-01,   1.84254005e-01,   3.47166288e-01,   9.16953877e-02,   3.81637525e-01,   
              3.53188856e-01,   3.43474263e-01,   3.75932899e-01])
  end

end 

end # module
