module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "MHD Multicomponent" begin

  @trixi_testset "elixir_mhdmulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
      l2   = [4.30029920e-02, 4.29875057e-02, 2.57471806e-02, 1.62185617e-01, 1.74536934e-02, 1.74545523e-02, 2.68731904e-02, 1.36464770e-15, 1.21243408e-02, 2.42486817e-02],
      linf = [3.13715220e-01, 3.03783978e-01, 2.15002288e-01, 9.04249573e-01, 9.39809810e-02, 9.47028202e-02, 1.52772540e-01, 8.24570183e-15, 7.87460541e-02, 1.57492108e-01])
  end

  @trixi_testset "elixir_mhdmulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_es.jl"),
      l2   = [4.25140100e-02, 4.25084765e-02, 2.38571816e-02, 1.15557310e-01, 1.63671989e-02, 1.63673846e-02, 2.58192976e-02, 2.34524850e-04, 1.08353280e-02, 2.16706560e-02],
      linf = [2.34580322e-01, 2.34685766e-01, 1.19133412e-01, 5.33372522e-01, 6.17688441e-02, 6.17909503e-02, 9.60217532e-02, 4.44580923e-03, 6.18887680e-02, 1.23777536e-01])
  end

  @trixi_testset "elixir_mhdmulti_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_convergence.jl"),
      l2   = [0.0003827494352191891, 0.0003827494352191414, 0.0005161253331853331, 0.0005704446900280551, 0.000443264790332803, 0.0004432647903328268, 0.0005073530103410933, 0.00038526540117132834, 7.438157880192512e-5, 0.00014876315760385024, 0.0002975263152077005],
      linf = [0.0013927512912668982, 0.0013927512912661499, 0.002597102723833799, 0.0016567175206610996, 0.0020103639634265758, 0.002010363963426798, 0.0025762126248859716, 0.0011800383681934948, 0.00028863972195533605, 0.0005772794439106721, 0.0011545588878213442])
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
