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
      l2   = [9.321181253877613e-7, 1.4181210743440328e-6, 1.4181210744199937e-6, 4.824553091402777e-6],
      linf = [9.577246540715123e-6, 1.1707525988002843e-5, 1.1707525988002843e-5, 4.886961560224279e-5])
  end

  @testset "elixir_euler_source_terms_rotated.jl" begin
    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.0" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [9.321181252866306e-7, 1.4181210743538732e-6, 1.4181210742955931e-6, 4.824553091287347e-6],
      linf = [9.577246546710327e-6, 1.1707525997328716e-5, 1.1707525993553958e-5, 4.886961556316294e-5],
      alpha = 0.0)
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.1" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [9.321188057029291e-7, 1.3195106906473365e-6, 1.510307360354032e-6, 4.82455408101712e-6],
      linf = [9.57723626271445e-6, 1.0480225511866337e-5, 1.2817828088262928e-5, 4.886962393513272e-5],
      alpha = 0.1)
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.2 * pi" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [9.32127973957391e-7, 8.477824799744325e-7, 1.8175286311402784e-6, 4.824562453521076e-6],
      linf = [9.576898420737834e-6, 5.057704352218195e-6, 1.635260719945464e-5, 4.886978754825577e-5],
      alpha = 0.2 * pi)
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.5 * pi" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [9.321181251295825e-7, 1.4181210742069683e-6, 1.4181210743242745e-6, 4.824553091237043e-6],
      linf = [9.57724653671832e-6, 1.1707525996440538e-5, 1.170752599466418e-5, 4.886961557826197e-5],
      alpha = 0.5 * pi)
    end
  end

  @trixi_testset "elixir_euler_source_terms_parallelogram.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_parallelogram.jl"),
      l2   = [1.1167802955144833e-5, 1.0805775514153104e-5, 1.953188337010932e-5, 5.5033856574857146e-5],
      linf = [8.297006495561199e-5, 8.663281475951301e-5, 0.00012264160606778596, 0.00041818802502024965])
  end

  @trixi_testset "elixir_euler_source_terms_waving_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_waving_flag.jl"),
      l2   = [2.991891317562739e-5, 3.6063177168283174e-5, 2.7082941743640572e-5, 0.00011414695350996946],
      linf = [0.0002437454930492855, 0.0003438936171968887, 0.00024217622945688078, 0.001266380414757684])
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
      l2   = [2.259440511901724e-6, 2.3188881559075347e-6, 2.3188881559568146e-6, 6.332786324137878e-6],
      linf = [1.4987382622067003e-5, 1.918201192063762e-5, 1.918201192019353e-5, 6.052671713430158e-5])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.03774907669925568, 0.02845190575242045, 0.028262802829412605, 0.13785915638851698],
      linf = [0.3368296929764073, 0.27644083771519773, 0.27990039685141377, 1.1971436487402016],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [3.69856202e-01, 2.35242180e-01, 2.41444928e-01, 1.28807120e+00],
      linf = [1.82786223e+00, 1.30452904e+00, 1.40347257e+00, 6.21791658e+00],
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
