module TestExamplesStructuredMesh2D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "structured_2d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "StructuredMesh2D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @trixi_testset "elixir_advection_extended.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [4.842990962468553e-6],
      linf = [3.47372094784415e-5])
  end

  @trixi_testset "elixir_advection_extended.jl with polydeg=4" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [2.0610527374594057e-6],
      linf = [7.21793425673134e-6],
      atol = 1e-12, # required to make CI tests pass on macOS
      cells_per_dimension = (16, 23),
      polydeg = 4,
      cfl = 1.4)
  end

  @testset "elixir_advection_rotated.jl" begin
    @trixi_testset "elixir_advection_rotated.jl with α = 0.0" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
        l2   = [7.013143474176369e-6],
        linf = [4.906526503622999e-5],
        alpha = 0.0)
    end

    @trixi_testset "elixir_advection_rotated.jl with α = 0.1" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
        l2   = [7.013143474176369e-6],
        linf = [4.906526503622999e-5],
        alpha = 0.1)
    end

    @trixi_testset "elixir_advection_rotated.jl with α = 0.5 * pi" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
        l2   = [7.013143474176369e-6],
        linf = [4.906526503622999e-5],
        alpha = 0.5 * pi)
    end
  end

  @trixi_testset "elixir_advection_parallelogram.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_parallelogram.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @trixi_testset "elixir_advection_waving_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_waving_flag.jl"),
      l2   = [0.00017007823031108628],
      linf = [0.0015264963372674245])
  end

  @trixi_testset "elixir_advection_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream.jl"),
      l2   = [5.984863383701255e-15],
      linf = [1.8207657603852567e-13])
  end

  @trixi_testset "elixir_advection_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
      l2   = [0.00023766972629056245],
      linf = [0.004142508319267935])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [6.398955192910044e-6],
      linf = [3.474337336717426e-5])
  end

  @trixi_testset "elixir_advection_restart.jl with waving flag mesh" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [0.00017274040834067234],
      linf = [0.0015435741643734513],
      elixir_file="elixir_advection_waving_flag.jl",
      restart_file="restart_000041.h5")
  end

  @trixi_testset "elixir_advection_restart.jl with free stream mesh" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [6.639325026542281e-15],
      linf = [1.829647544582258e-13],
      elixir_file="elixir_advection_free_stream.jl",
      restart_file="restart_000068.h5")
  end

  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5])
  end

  @testset "elixir_euler_source_terms_rotated.jl" begin
    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.0" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5],
      alpha = 0.0)
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.1" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [8.517816067144339e-7, 1.1545335192659009e-6, 1.3105743360575469e-6, 4.2778880887284874e-6],
      linf = [8.357837601113971e-6, 9.243793987812055e-6, 1.1305611891110345e-5, 4.496194865932779e-5],
      alpha = 0.1)
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.2 * pi" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [8.517916398386897e-7, 7.775434972297097e-7, 1.5639649498758719e-6, 4.277897693664215e-6],
      linf = [8.357487707666422e-6, 4.287882448716918e-6, 1.4423290043641401e-5, 4.496211536153538e-5],
      alpha = 0.2 * pi)
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.5 * pi" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5],
      alpha = 0.5 * pi)
    end
  end

  @trixi_testset "elixir_euler_source_terms_parallelogram.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_parallelogram.jl"),
      l2   = [1.0893247337495366e-5, 1.0425688625462763e-5, 1.755222105014883e-5, 5.136512290929069e-5],
      linf = [7.449791413693951e-5, 7.621073881791673e-5, 0.00011093303834863733, 0.00039625209916493986])
  end

  @trixi_testset "elixir_euler_source_terms_waving_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_waving_flag.jl"),
      l2   = [3.2280277059090205e-5, 3.481614479752735e-5, 2.8784017747658748e-5, 0.00011549476000734391],
      linf = [0.00025339087459608223, 0.0003425481056145152, 0.0002454647901921625, 0.0012806891514367535])
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [2.063350241405049e-15, 1.8571016296925367e-14, 3.1769447886391905e-14, 1.4104095258528071e-14],
      linf = [1.9539925233402755e-14, 2.9791447087035294e-13, 6.502853810985698e-13, 2.7000623958883807e-13],
      atol = 7.0e-13)
  end

  @trixi_testset "elixir_euler_free_stream.jl with FluxRotated(flux_lax_friedrichs)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      surface_flux=FluxRotated(flux_lax_friedrichs),
      l2   = [2.063350241405049e-15, 1.8571016296925367e-14, 3.1769447886391905e-14, 1.4104095258528071e-14],
      linf = [1.9539925233402755e-14, 2.9791447087035294e-13, 6.502853810985698e-13, 2.7000623958883807e-13],
      atol = 7.0e-13)
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [2.3653424742684444e-6, 2.1388875095440695e-6, 2.1388875095548492e-6, 6.010896863397195e-6],
      linf = [1.4080465931654018e-5, 1.7579850587257084e-5, 1.7579850592586155e-5, 5.956893531156027e-5])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.03774907669925568, 0.02845190575242045, 0.028262802829412605, 0.13785915638851698],
      linf = [0.3368296929764073, 0.27644083771519773, 0.27990039685141377, 1.1971436487402016],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_euler_rayleigh_taylor_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_rayleigh_taylor_instability.jl"),
      l2   = [0.06365630381017849, 0.007166887387738937, 0.002878708825497772, 0.010247678114070121],
      linf = [0.4799214336153155, 0.024595483032220266, 0.02059808120543466, 0.03190756362943725],
      cells_per_dimension = (8,8),
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
      l2   = [0.8799744480157664, 0.8535008397034816, 0.7851383019164209],
      linf = [1.0771947577311836, 1.9143913544309838, 2.149549109115789],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_harmonic_nonperiodic.jl"),
      l2   = [0.19357947606509474, 0.47041398037626814, 0.4704139803762686],
      linf = [0.35026352556630114, 0.8344372248051408, 0.8344372248051408],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.04937480811868297, 0.06117033019988596, 0.060998028674664716, 0.03155145889799417,
              0.2319175391388658, 0.02476283192966346, 0.024483244374818587, 0.035439957899127385,
              0.0016022148194667542],
      linf = [0.24749024430983746, 0.2990608279625713, 0.3966937932860247, 0.22265033744519683,
              0.9757376320946505, 0.12123736788315098, 0.12837436699267113, 0.17793825293524734,
              0.03460761690059514],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.02890769490562535, 0.0062599448721613205, 0.005650300017676721, 0.007334415940022972,
              0.00490446035599909, 0.007202284100220619, 0.007003258686714405, 0.006734267830082687,
              0.004253003868791559],
      linf = [0.17517380432288565, 0.06197353710696667, 0.038494840938641646, 0.05293345499813148,
              0.03817506476831778, 0.042847170999492534, 0.03761563456810613, 0.048184237474911844,
              0.04114666955364693],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
      l2   = [0.0017285599436729316, 0.025584610912606776, 0.028373834961180594, 6.274146767730866e-5],
      linf = [0.012972309788264802, 0.108283714215621, 0.15831585777928936, 0.00018196759554722775],
      tspan = (0.0, 0.05))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # module
