module TestExamples1DEuler

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_1d_dgsem")

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
      l2   = [0.0011482554820185795, 0.00011482554830363504, 5.741277417754598e-6],
      linf = [0.004090978306820037, 0.00040909783134346345, 2.0454891732413216e-5])
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
      l2   = [0.11915540925414216, 0.15489191247295198, 0.44543052524765375],
      linf = [0.2751485868543495, 0.2712764982000735, 0.9951407418216425])
  end

  @trixi_testset "elixir_euler_ec.jl with flux_kennedy_gruber" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07905582221868049, 0.10180958900546237, 0.29596551476711125],
      linf = [0.23515297345769826, 0.2958208108392532, 0.8694224308790321],
      maxiters = 10,
      surface_flux = flux_kennedy_gruber,
      volume_flux = flux_kennedy_gruber)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_shima_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07909267609417114, 0.1018246500951966, 0.2959649187481973],
      linf = [0.23631829743146504, 0.2977756307879202, 0.8642794698697331],
      maxiters = 10,
      surface_flux = flux_shima_etal,
      volume_flux = flux_shima_etal)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07905306555214126, 0.10181180378499956, 0.2959171937479504],
      linf = [0.24057642004451651, 0.29691454643616433, 0.886425723870524],
      maxiters = 10,
      surface_flux = flux_chandrashekar,
      volume_flux = flux_chandrashekar)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.07959780803600519, 0.10342491934977621, 0.2978851659149904],
      linf = [0.19228754121840885, 0.2524152253292552, 0.725604944702432],
      maxiters = 10,
      surface_flux = flux_hll,
      volume_flux = flux_ranocha)
  end

  @trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.11665968950973675, 0.15105507394693413, 0.43503082674771115],
      linf = [0.1867400345208743, 0.24621854448555328, 0.703826406555577])
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [1.250005061244617, 0.06878411345533507, 0.9264328311018613],
      linf = [2.9766770877037168, 0.16838100902295852, 2.6655773445485798],
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_pure_fv.jl"),
      l2   = [1.075075094036344, 0.06766902169711514, 0.9221426570128292],
      linf = [3.3941512671408542, 0.16862631133303882, 2.6572394126490315],
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
      l2   = [0.21651329948737183, 0.28091709900008616, 0.5580778880050432],
      linf = [1.513525457073142, 1.5328754303137992, 2.0467706106669556],
      maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_neuralnetwork_perssonperaire.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_neuralnetwork_perssonperaire.jl"),
        l2   = [2.13605618e-01, 2.79953055e-01, 5.54424459e-01],
        linf = [1.55151701e+00, 1.55696782e+00, 2.05525953e+00],
        maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_neuralnetwork_rayhesthaven.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_neuralnetwork_rayhesthaven.jl"),
        l2   = [2.18148857e-01, 2.83182959e-01, 5.59096194e-01],
        linf = [1.62706876e+00, 1.61680275e+00, 2.05876517e+00],
        maxiters = 30)
  end
end

end # module
