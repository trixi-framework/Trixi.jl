module TestExamples3DEuler

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi.jl/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_3d_dgsem")

@testset "Compressible Euler" begin
  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.010385936842224346, 0.009776048833895767, 0.00977604883389591, 0.009776048833895733, 0.01506687097416608],
      linf = [0.03285848350791731, 0.0321792316408982, 0.032179231640894645, 0.032179231640895534, 0.0655408023333299])
  end

  @trixi_testset "elixir_euler_convergence_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence_pure_fv.jl"),
      l2   = [0.037182410351406,  0.032062252638283974, 0.032062252638283974, 0.03206225263828395,  0.12228177813586687],
      linf = [0.0693648413632646, 0.0622101894740843,   0.06221018947408474,  0.062210189474084965, 0.24196451799555962])
  end

  @trixi_testset "elixir_euler_source_terms.jl with split_form" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.010385936842223388, 0.009776048833894784, 0.009776048833894784, 0.009776048833894765, 0.015066870974164096],
      linf = [0.03285848350791687, 0.032179231640897754, 0.0321792316408942, 0.0321792316408982, 0.06554080233333615],
      volume_integral=VolumeIntegralFluxDifferencing(flux_central))
  end

  @trixi_testset "elixir_euler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
      l2   = [0.0003637241020254405, 0.0003955570866382718, 0.0003955570866383613, 0.00039555708663834417, 0.0007811613481640202],
      linf = [0.0024000660244674066, 0.0029635410025339315, 0.0029635410025292686, 0.002963541002525938, 0.007191437359396424])
  end

  @trixi_testset "elixir_euler_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_mortar.jl"),
      l2   = [0.0019428114665068841, 0.0018659907926698422, 0.0018659907926698589, 0.0018659907926698747, 0.0034549095578444056],
      linf = [0.011355360771142298, 0.011526889155693887, 0.011526889155689002, 0.011526889155701436, 0.02299726519821288])
  end

  @trixi_testset "elixir_euler_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_amr.jl"),
      l2   = [0.0038281920613404716, 0.003828192061340465, 0.0038281920613404694, 0.0038281920613404672, 0.005742288092010652],
      linf = [0.07390396464027349, 0.07390396464027305, 0.07390396464027305, 0.07390396464027305, 0.11085594696041134],
      tspan=(0.0, 0.1),
      coverage_override = (maxiters=6, initial_refinement_level=0, base_level=0, med_level=0, max_level=1))
  end

  @trixi_testset "elixir_euler_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      l2   = [0.00034949871748737876, 0.03133384111621587, 0.03133384111621582, 0.04378599329988925, 0.015796137903453026],
      linf = [0.0013935237751798724, 0.0724080091006194, 0.07240800910061806, 0.12795921224174792, 0.07677156293692633],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.02570137197844877, 0.016179934130642552, 0.01617993413064253, 0.016172648598753545, 0.09261669328795467],
      linf = [0.3954458125573179, 0.26876916180359345, 0.26876916180359345, 0.26933123042178553, 1.3724137121660251])
  end

  @trixi_testset "elixir_euler_shockcapturing_amr.jl" begin
    # OBS! This setup does not make much practical sense. It is only added to exercise the
    # `sedov_self_gravity` AMR indicator, which in its original configuration is too expensive for
    # CI testing
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_amr.jl"),
      l2   = [0.02217299067704248, 0.012771561294571411, 0.01277156129457143, 0.012770635779336643, 0.08091898488262424],
      linf = [0.4047819603427084, 0.27493532130155474, 0.2749353213015551, 0.2749304638368023, 1.4053942765487641],
      maxiters=10,
      coverage_override = (maxiters=2,))
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
      tspan = (0.0, 0.2),
      # Let this test run longer to cover some lines in the positivity preserving limiter
      # and some AMR lines
      coverage_override = (maxiters=10^5,))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.0007127163978031706, 0.0023166296394624025, 0.002316629639462401, 0.0023166296394624038, 0.010200581509653256],
      linf = [0.06344190883105805, 0.6292607955969378, 0.6292607955969377, 0.6292607955969377, 2.397746252817731],
      maxiters=5, max_level=6,
      coverage_override = (maxiters=2, initial_refinement_level=1, base_level=1, max_level=3))
  end
end

end # module
