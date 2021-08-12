module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "MHD Multicomponent" begin

  @trixi_testset "elixir_mhdmulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
      l2   = [4.30105889e-02, 4.29956523e-02, 2.57513367e-02, 1.62116945e-01, 1.74486380e-02, 1.74489235e-02, 2.68694627e-02, 2.35777594e-04, 1.21212952e-02, 2.42425903e-02],
      linf = [3.08924226e-01, 3.07301938e-01, 2.18934142e-01, 9.43237292e-01, 9.18013046e-02, 9.25436277e-02, 1.69600283e-01, 7.96105863e-03, 7.96514719e-02, 1.59302944e-01])
  end

  @trixi_testset "elixir_mhdmulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_es.jl"),
      l2   = [4.25140100e-02, 4.25084765e-02, 2.38571816e-02, 1.15557310e-01, 1.63671989e-02, 1.63673846e-02, 2.58192976e-02, 2.34524850e-04, 1.08353280e-02, 2.16706560e-02],
      linf = [2.34580322e-01, 2.34685766e-01, 1.19133412e-01, 5.33372522e-01, 6.17688441e-02, 6.17909503e-02, 9.60217532e-02, 4.44580923e-03, 6.18887680e-02, 1.23777536e-01])
  end

  @trixi_testset "elixir_mhdmulti_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_convergence.jl"),
      l2   = [0.00026086814345575545, 0.0002608681434558476, 0.00048401141419704093, 0.0003246315763252851, 0.0003655995669606298, 0.00036559956696067316, 0.0005007680779079482, 0.0003855610470758256, 5.258783225290134e-5, 0.00010517566450580267, 0.00021035132901160535],
      linf = [0.0015430152095171022, 0.0015430152095179052, 0.0023883101781443854, 0.0010097026785059748, 0.001880514662696009, 0.00188051466269612, 0.0026096395828504393, 0.0010868614907852264, 0.0002056455895989573, 0.0004112911791979146, 0.0008225823583958292])
  end

  @trixi_testset "elixir_mhdmulti_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_rotor.jl"),
      l2   = [7.17665548e-01, 7.22954510e-01, 0.00000000e+00, 8.34512452e-01, 4.74961443e-02, 7.91093545e-02, 0.00000000e+00, 1.98437701e-03, 1.91605624e-01, 9.58028120e-02],
      linf = [1.11390901e+01, 1.10181050e+01, 0.00000000e+00, 1.24464832e+01, 6.35439011e-01, 1.06796978e+00, 0.00000000e+00, 6.41596561e-02, 3.49017111e+00, 1.74508556e+00],
      tspan = (0.0, 0.01))

  # TODO: nonconservative terms, remove
  @trixi_testset "elixir_mhdmulti_ec.jl with old nonconservative stuff" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
      l2   = [4.30029920e-02, 4.29875057e-02, 2.57471806e-02, 1.62185617e-01, 1.74536934e-02, 1.74545523e-02, 2.68731904e-02, 1.36052812e-15, 1.21243408e-02, 2.42486817e-02],
      linf = [3.13715220e-01, 3.03783978e-01, 2.15002288e-01, 9.04249573e-01, 9.39809810e-02, 9.47028202e-02, 1.52772540e-01, 9.36599914e-15, 7.87460541e-02, 1.57492108e-01],
      volume_flux  = flux_hindenlang_gassner,
      surface_flux = flux_hindenlang_gassner)
  end
end

end

end # module
