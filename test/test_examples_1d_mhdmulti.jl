module TestExamples1DMHD

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

@testset "MHD Multicomponent" begin

  @trixi_testset "elixir_mhdmulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
      l2   = [1.69266396e-01, 3.04958381e-01, 0.00000000e+00, 3.45735220e-01, 6.60825891e-17, 3.17662584e-01, 0.00000000e+00, 2.18373660e-02, 4.36747319e-02, 8.73494638e-02],
      linf = [5.08655909e-01, 1.25176699e+00, 0.00000000e+00, 1.99579851e+00, 1.11022302e-16, 1.68777938e+00, 0.00000000e+00, 1.00173010e-01, 2.00346020e-01, 4.00692040e-01])
  end

  @trixi_testset "elixir_mhdmulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_es.jl"),
      l2   = [1.66364836e-01, 2.87812904e-01, 0.00000000e+00, 3.17156609e-01, 6.60825891e-17, 3.16727982e-01, 0.00000000e+00, 1.87387435e-02, 3.74774869e-02, 7.49549738e-02],
      linf = [5.25859519e-01, 1.11130058e+00, 0.00000000e+00, 1.46798145e+00, 1.11022302e-16, 1.86689645e+00, 0.00000000e+00, 7.63433974e-02, 1.52686795e-01, 3.05373590e-01])
  end

  @trixi_testset "elixir_mhdmulti_eoc.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_eoc.jl"),
      l2   = [1.70884468e-05, 3.33778117e-04, 3.33778117e-04, 3.60816702e-05, 4.13046273e-17, 3.33548882e-04, 3.33548882e-04, 1.30233539e-05, 2.60467077e-05, 5.20934154e-05],
      linf = [2.34173129e-05, 1.23807753e-03, 1.23807753e-03, 1.07512068e-04, 1.11022302e-16, 1.22960686e-03, 1.22960686e-03, 2.01157231e-05, 4.02314461e-05, 8.04628923e-05])
    end

  @trixi_testset "elixir_mhdmulti_briowu_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_briowu_shock_tube.jl"),
      l2   = [1.93221943e-01, 3.56599349e-01, 0.00000000e+00, 3.63446684e-01, 6.33435897e-16, 3.66051482e-01, 0.00000000e+00, 5.65287156e-02, 1.13057431e-01],
      linf = [4.33927958e-01, 1.09948372e+00, 0.00000000e+00, 1.03943772e+00, 2.99760217e-15, 1.50960600e+00, 0.00000000e+00, 1.96228180e-01, 3.92456360e-01])
    end

end

end # module
