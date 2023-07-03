module TestExamples1DEuler

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Compressible Euler" begin
  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [2.2527950196212703e-8, 1.8187357193835156e-8, 7.705669939973104e-8],
      linf = [1.6205433861493646e-7, 1.465427772462391e-7, 5.372255111879554e-7])
  end

  @trixi_testset "elixir_euler_convergence_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence_pure_fv.jl"),
      l2   = [0.019355699748523896, 0.022326984561234497, 0.02523665947241734],
      linf = [0.02895961127645519,  0.03293442484199227,  0.04246098278632804])
  end

  @trixi_testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.0011482554820217855, 0.00011482554830323462, 5.741277429325267e-6],
      linf = [0.004090978306812376, 0.0004090978313582294, 2.045489210189544e-5])
  end

  @trixi_testset "elixir_euler_density_wave.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [7.71293052584723e-16, 1.9712947511091717e-14, 7.50672833504266e-15],
      linf = [3.774758283725532e-15, 6.733502644351574e-14, 2.4868995751603507e-14],
      initial_condition = initial_condition_constant)
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [3.8099996914101204e-6, 1.6745575717106341e-6, 7.732189531480852e-6],
      linf = [1.2971473393186272e-5, 9.270328934274374e-6, 3.092514399671842e-5])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.11821957357197649, 0.15330089521538678, 0.4417674632047301],
      linf = [0.24280567569982958, 0.29130548795961936, 0.8847009003152442])
  end

  @trixi_testset "elixir_euler_ec.jl with flux_kennedy_gruber" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07803455838661963, 0.10032577312032283, 0.29228156303827935],
      linf = [0.2549869853794955, 0.3376472164661263, 0.9650477546553962],
      maxiters = 10,
      surface_flux = flux_kennedy_gruber,
      volume_flux = flux_kennedy_gruber)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_shima_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07800654460172655, 0.10030365573277883, 0.2921481199111959],
      linf = [0.25408579350400395, 0.3388657679031271, 0.9776486386921928],
      maxiters = 10,
      surface_flux = flux_shima_etal,
      volume_flux = flux_shima_etal)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07801923089205756, 0.10039557434912669, 0.2922210399923278],
      linf = [0.2576521982607225, 0.3409717926625057, 0.9772961936567048],
      maxiters = 10,
      surface_flux = flux_chandrashekar,
      volume_flux = flux_chandrashekar)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07852272782240548, 0.10209790867523805, 0.293873048809011],
      linf = [0.19244768908604093, 0.2515941686151897, 0.7258000837553769],
      maxiters = 10,
      surface_flux = flux_hll,
      volume_flux = flux_ranocha)
  end

  @trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.11606096465319675, 0.15028768943458806, 0.4328230323046703],
      linf = [0.18031710091067965, 0.2351582421501841, 0.6776805692092567])
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [1.250005061244617, 0.06878411345533507, 0.9264328311018613],
      linf = [2.9766770877037168, 0.16838100902295852, 2.6655773445485798],
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_pure_fv.jl"),
      l2   = [1.0735456065491455, 0.07131078703089379, 0.9205739468590453],
      linf = [3.4296365168219216, 0.17635583964559245, 2.6574584326179505],
      # Let this test run longer to cover some lines in flux_hllc
      coverage_override = (maxiters=10^5, tspan=(0.0, 0.1)))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl with pressure" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [1.297525985166995, 0.07964929522694145, 0.9269991156246368],
      linf = [3.1773015255764427, 0.21331831536493773, 2.6650170188241047],
      shock_indicator_variable = pressure,
      cfl = 0.2,
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl with density" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [1.2798798835860528, 0.07103461242058921, 0.9273792517187003],
      linf = [3.1087017048015824, 0.17734706962928956, 2.666689753470263],
      shock_indicator_variable = density,
      cfl = 0.2,
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_euler_positivity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_positivity.jl"),
      l2   = [1.6493820253458906, 0.19793887460986834, 0.9783506076125921],
      linf = [4.71751203912051, 0.5272411022735763, 2.7426163947635844],
      coverage_override = (maxiters=3,))
  end

  @trixi_testset "elixir_euler_blast_wave.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [0.21934822867340323, 0.28131919126002686, 0.554361702716662],
      linf = [1.5180897390290355, 1.3967085956620369, 2.0663825294019595],
      maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_neuralnetwork_perssonperaire.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_neuralnetwork_perssonperaire.jl"),
        l2   = [0.21814833203212694, 0.2818328665444332, 0.5528379124720818],
        linf = [1.5548653877320868, 1.4474018998129738, 2.071919577393772],
        maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_neuralnetwork_rayhesthaven.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_neuralnetwork_rayhesthaven.jl"),
        l2   = [0.22054468879127423, 0.2828269190680846, 0.5542369885642424],
        linf = [1.5623359741479623, 1.4290121654488288, 2.1040405133123072],
        maxiters = 30)
  end
end

end # module
