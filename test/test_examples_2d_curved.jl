module TestExamples2DCurved

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Curved Mesh" begin
  @testset "elixir_advection_basic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_curved.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @testset "elixir_advection_extended_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended_curved.jl"),
      l2   = [4.842990962468553e-6],
      linf = [3.47372094784415e-5])
  end

  @testset "elixir_advection_extended_curved.jl with polydeg=4" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended_curved.jl"),
      l2   = [2.0610527374594057e-6],
      linf = [7.21793425673134e-6],
      atol = 1e-12, # required to make CI tests pass on macOS
      cells_per_dimension = (16, 23),
      polydeg = 4,
      cfl = 1.4)
  end

  @testset "elixir_advection_rotated.jl" begin
    @testset "elixir_advection_rotated.jl with α = 0.0" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
        l2   = [7.013143474176369e-6],
        linf = [4.906526503622999e-5],
        alpha = 0.0)
    end

    @testset "elixir_advection_rotated.jl with α = 0.1" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
        l2   = [7.013143474176369e-6],
        linf = [4.906526503622999e-5],
        alpha = 0.1)
    end

    @testset "elixir_advection_rotated.jl with α = 0.5 * pi" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
        l2   = [7.013143474176369e-6],
        linf = [4.906526503622999e-5],
        alpha = 0.5 * pi)
    end
  end

  @testset "elixir_advection_parallelogram.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_parallelogram.jl"),
      l2   = [0.0005165995033861579],
      linf = [0.002506176163321605])
  end

  @testset "elixir_advection_waving_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_waving_flag.jl"),
      l2   = [0.00017007823031108628],
      linf = [0.0015264963372674245])
  end

  @testset "elixir_advection_free_stream_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream_curved.jl"),
      l2   = [5.984863383701255e-15],
      linf = [1.8207657603852567e-13])
  end

  @testset "elixir_advection_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic_curved.jl"),
      l2   = [0.00023766972629056245],
      linf = [0.004142508319267935])
  end

  @testset "elixir_advection_restart_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_curved.jl"),
      l2   = [6.398955192910044e-6],
      linf = [3.474337336717426e-5])
  end

  @testset "elixir_advection_restart_curved.jl with waving flag mesh" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_curved.jl"),
      l2   = [0.00017274040834067234],
      linf = [0.0015435741643734513],
      elixir_file="elixir_advection_waving_flag.jl",
      restart_file="restart_000041.h5")
  end

  @testset "elixir_advection_restart_curved.jl with free stream mesh" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_curved.jl"),
      l2   = [6.639325026542281e-15],
      linf = [1.829647544582258e-13],
      elixir_file="elixir_advection_free_stream_curved.jl",
      restart_file="restart_000068.h5")
  end

  @testset "elixir_euler_source_terms_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_curved.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5])
  end

  @testset "elixir_euler_source_terms_rotated.jl" begin
    @testset "elixir_euler_source_terms_rotated.jl with α = 0.0" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5],
      alpha = 0.0)
    end

    @testset "elixir_euler_source_terms_rotated.jl with α = 0.1" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [8.517816067144339e-7, 1.1545335192659009e-6, 1.3105743360575469e-6, 4.2778880887284874e-6],
      linf = [8.357837601113971e-6, 9.243793987812055e-6, 1.1305611891110345e-5, 4.496194865932779e-5],
      alpha = 0.1)
    end

    @testset "elixir_euler_source_terms_rotated.jl with α = 0.2 * pi" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [8.517916398386897e-7, 7.775434972297097e-7, 1.5639649498758719e-6, 4.277897693664215e-6],
      linf = [8.357487707666422e-6, 4.287882448716918e-6, 1.4423290043641401e-5, 4.496211536153538e-5],
      alpha = 0.2 * pi)
    end

    @testset "elixir_euler_source_terms_rotated.jl with α = 0.5 * pi" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_rotated.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5],
      alpha = 0.5 * pi)
    end
  end

  @testset "elixir_euler_source_terms_parallelogram.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_parallelogram.jl"),
      l2   = [1.0893247337495366e-5, 1.0425688625462763e-5, 1.755222105014883e-5, 5.136512290929069e-5],
      linf = [7.449791413693951e-5, 7.621073881791673e-5, 0.00011093303834863733, 0.00039625209916493986])
  end

  @testset "elixir_euler_source_terms_waving_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_waving_flag.jl"),
      l2   = [3.2280277059090205e-5, 3.481614479752735e-5, 2.8784017747658748e-5, 0.00011549476000734391],
      linf = [0.00025339087459608223, 0.0003425481056145152, 0.0002454647901921625, 0.0012806891514367535])
  end

  @testset "elixir_euler_free_stream_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_curved.jl"),
      l2   = [2.063350241405049e-15, 1.8571016296925367e-14, 3.1769447886391905e-14, 1.4104095258528071e-14],
      linf = [1.9539925233402755e-14, 2.9791447087035294e-13, 6.502853810985698e-13, 2.7000623958883807e-13])
  end

  @testset "elixir_euler_free_stream_curved.jl with FluxRotated(flux_lax_friedrichs)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_curved.jl"),
      surface_flux=FluxRotated(flux_lax_friedrichs),
      l2   = [2.063350241405049e-15, 1.8571016296925367e-14, 3.1769447886391905e-14, 1.4104095258528071e-14],
      linf = [1.9539925233402755e-14, 2.9791447087035294e-13, 6.502853810985698e-13, 2.7000623958883807e-13])
  end

  @testset "elixir_euler_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic_curved.jl"),
      l2   = [2.3653424742684444e-6, 2.1388875095440695e-6, 2.1388875095548492e-6, 6.010896863397195e-6],
      linf = [1.4080465931654018e-5, 1.7579850587257084e-5, 1.7579850592586155e-5, 5.956893531156027e-5])
  end

  @testset "elixir_euler_ec_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec_curved.jl"),
      l2   = [0.03774170146315357, 0.028439692043402822, 0.028270724308772303, 0.13784229328899064],
      linf = [0.33571598859050134, 0.27831002252333553, 0.295995432205428, 1.2178776811996832],
      tspan = (0.0, 0.3))
  end

  @testset "elixir_hypdiff_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic_curved.jl"),
      l2   = [0.8799744480157664, 0.8535008397034816, 0.7851383019164209],
      linf = [1.0771947577311836, 1.9143913544309838, 2.149549109115789],
      tspan = (0.0, 0.1))
  end

end

end # module
