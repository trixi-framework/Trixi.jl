module TestExamples3DEuler

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_3d_dgsem")

@testset "Compressible Euler" begin
  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.01032310150257373, 0.009728768969448439, 0.009728768969448494, 0.009728768969448388, 0.015080412597559597],
      linf = [0.034894790428615874, 0.033835365548322116, 0.033835365548322116, 0.03383536554832034, 0.06785765131417065])
  end

  @trixi_testset "elixir_euler_convergence_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence_pure_fv.jl"),
      l2   = [0.037182410351406,  0.032062252638283974, 0.032062252638283974, 0.03206225263828395,  0.12228177813586687],
      linf = [0.0693648413632646, 0.0622101894740843,   0.06221018947408474,  0.062210189474084965, 0.24196451799555962])
  end

  @trixi_testset "elixir_euler_source_terms.jl with split_form" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.010323101502574773, 0.009728768969449282, 0.009728768969449344, 0.009728768969449401, 0.015080412597560888],
      linf = [0.0348947904286212, 0.03383536554832034, 0.03383536554831723, 0.03383536554831679, 0.0678576513141671],
      volume_integral=VolumeIntegralFluxDifferencing(flux_central))
  end

  @trixi_testset "elixir_euler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
      l2   = [0.0003637241020254405, 0.0003955570866382718, 0.0003955570866383613, 0.00039555708663834417, 0.0007811613481640202],
      linf = [0.0024000660244674066, 0.0029635410025339315, 0.0029635410025292686, 0.002963541002525938, 0.007191437359396424])
  end

  @trixi_testset "elixir_euler_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_mortar.jl"),
      l2   = [0.0019011097544691046, 0.0018289464161846331, 0.0018289464161847266, 0.0018289464161847851, 0.0033547668596639966],
      linf = [0.011918626829790169, 0.011808582902362641, 0.01180858290237552, 0.011808582902357312, 0.024648094686513744])
  end

  @trixi_testset "elixir_euler_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_amr.jl"),
      l2   = [0.0038281919675174224, 0.003828191967517416, 0.003828191967517415, 0.003828191967517416, 0.0057422879512759525],
      linf = [0.07390148817126874, 0.07390148817126896, 0.07390148817126896, 0.07390148817126874, 0.110852232256903],
      tspan=(0.0, 0.1))
  end

  @trixi_testset "elixir_euler_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      l2   = [0.000349497725802709, 0.031333864159293845, 0.03133386415929371, 0.04378595044851934, 0.015796652303527357],
      linf = [0.0013934750216293423, 0.07242017454880123, 0.07242017454880156, 0.12796560115483002, 0.07680757651078807],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.02570137197844877, 0.016179934130642552, 0.01617993413064253, 0.016172648598753545, 0.09261669328795467],
      linf = [0.3954458125573179, 0.26876916180359345, 0.26876916180359345, 0.26933123042178553, 1.3724137121660251])
  end

  @trixi_testset "elixir_euler_shockcapturing.jl with initial_condition_sedov_self_gravity" begin
    # OBS! This setup does not run longer but crashes (also the parameters do not make sense) -> only for testing the IC!
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.048222481449477446, 0.0517877776783649, 0.051787777678364795, 0.05178777767836493, 0.230846733492145],
      linf = [0.954296651568088, 1.2654683413909855, 1.2654683413909855, 1.2654683413909857, 12.805752164822744],
      initial_condition=initial_condition_sedov_self_gravity, cfl=0.7, alpha_max=1.0, tspan=(0.0, 0.1))
  end

  @trixi_testset "elixir_euler_shockcapturing_amr.jl" begin
    # OBS! This setup does not make much practical sense. It is only added to exercise the
    # `sedov_self_gravity` AMR indicator, which in its original configuration is too expensive for
    # CI testing
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_amr.jl"),
      l2   = [0.02217299067704248, 0.012771561294571411, 0.01277156129457143, 0.012770635779336643, 0.08091898488262424],
      linf = [0.4047819603427084, 0.27493532130155474, 0.2749353213015551, 0.2749304638368023, 1.4053942765487641],
      maxiters=10)
  end

  @trixi_testset "elixir_euler_density_pulse.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_pulse.jl"),
      l2   = [0.057196526814004715, 0.057196526814004715, 0.05719652681400473, 0.057196526814004736, 0.08579479022100575],
      linf = [0.27415246703018203, 0.2741524670301829, 0.2741524670301827, 0.27415246703018226, 0.41122870054527816])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.02526341317987378, 0.016632068583699623, 0.016632068583699623, 0.01662548715216875, 0.0913477018048886],
      linf = [0.4372549540810414, 0.28613118232798984, 0.28613118232799006, 0.28796686065271876, 1.5072828647309124])
  end

  @trixi_testset "elixir_euler_ec.jl with initial_condition=initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [4.183721551616214e-16, 6.059779958716338e-16, 4.916596221090319e-16, 9.739943366304456e-16, 3.7485908743251566e-15],
      linf = [2.4424906541753444e-15, 3.733124920302089e-15, 4.440892098500626e-15, 5.329070518200751e-15, 2.4868995751603507e-14],
      initial_condition=initial_condition_constant)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.025265721172813106, 0.016649800693500427, 0.01664980069350042, 0.01664379306708522, 0.09137248646784184],
      linf = [0.4373399329742198, 0.28434487167605427, 0.28434487167605427, 0.28522678968890774, 1.532471676033761],
      surface_flux=flux_chandrashekar, volume_flux=flux_chandrashekar)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_kennedy_gruber" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.025280033869871984, 0.016675487948639846, 0.016675487948639853, 0.016668992714991282, 0.091455613470441],
      linf = [0.43348628145015766, 0.28853549062014217, 0.28853549062014217, 0.2903943042772536, 1.5236557526482426],
      surface_flux=flux_kennedy_gruber, volume_flux=flux_kennedy_gruber)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_shima_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.025261716925811403, 0.016637655557848952, 0.01663765555784895, 0.01663105921013437, 0.09136239054024566],
      linf = [0.43692416928732536, 0.28622033209064734, 0.28622033209064746, 0.2881197143457632, 1.506534270303663],
      surface_flux=flux_shima_etal, volume_flux=flux_shima_etal)
  end

  @trixi_testset "elixir_euler_blob_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_amr.jl"),
      l2   = [0.04867856452253151, 0.2640486962336911, 0.0354927658652858, 0.03549276586528571, 1.0777274757408568],
      linf = [9.558543313792217, 49.4518309553356, 10.319859082570309, 10.319859082570487, 195.1066220797401],
      tspan = (0.0, 0.2))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.0007127163978031706, 0.0023166296394624025, 0.002316629639462401, 0.0023166296394624038, 0.010200581509653256],
      linf = [0.06344190883105805, 0.6292607955969378, 0.6292607955969377, 0.6292607955969377, 2.397746252817731],
      maxiters=5, max_level=6)
  end
end

end # module
