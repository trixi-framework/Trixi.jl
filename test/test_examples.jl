using Test
import Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = joinpath(@__DIR__, "out")
isdir(outdir) && rm(outdir, recursive=true)

# Run basic tests
@testset "Examples (short execution time)" begin
  @testset "../examples/parameters.toml" begin
    test_trixi_run("../examples/parameters.toml",
            l2   = [9.144681765639205e-6],
            linf = [6.437440532547356e-5])
  end
  @testset "../examples/parameters_alfven_wave.toml" begin
    test_trixi_run("../examples/parameters_alfven_wave.toml",
            l2   = [0.00011134513490658689, 5.880188909157728e-6, 5.880188909159547e-6, 8.432880997656317e-6, 1.2942387343501909e-6, 1.2238820298971968e-6, 1.2238820298896402e-6, 1.830621754702352e-6, 8.086996786269551e-7],
            linf = [0.00025632790161078667, 1.6379021163651086e-5, 1.637902116437273e-5, 2.58759953227633e-5, 5.327732286231068e-6, 8.118520269495555e-6, 8.118520269606577e-6, 1.2107354757678879e-5, 4.1737070057713136e-6])
  end
  @testset "../examples/parameters_amr.toml" begin
    test_trixi_run("../examples/parameters_amr.toml",
            l2   = [0.12533080510721514],
            linf = [0.999980298294775])
  end
  @testset "../examples/parameters_amr_nonperiodic.toml" begin
    test_trixi_run("../examples/parameters_amr_nonperiodic.toml",
            l2   = [2.401816280408703e-5],
            linf = [0.0003102822951955575])
  end
  @testset "../examples/parameters_amr_vortex.toml" begin
    test_trixi_run("../examples/parameters_amr_vortex.toml",
            l2   = [2.0750351586876505e-6, 0.003281637561081054, 0.0032807189382436106, 0.0046470466205649425],
            linf = [4.625172721961501e-5, 0.0318570623352572, 0.031910329823320094, 0.04575283708569344])
  end
  @testset "../examples/parameters_blast_wave_shockcapturing.toml" begin
    test_trixi_run("../examples/parameters_blast_wave_shockcapturing.toml",
            l2   = [0.13910202327088322, 0.11538722576277083, 0.1153873048510009, 0.3387876385945495],
            linf = [1.454418325889352, 1.3236875559310013, 1.323687555933169, 1.8225476335086368])
  end
  @testset "../examples/parameters_ec.toml" begin
    test_trixi_run("../examples/parameters_ec.toml",
            l2   = [0.06159341742582756, 0.05012484425381723, 0.05013298724507752, 0.22537740506116724],
            linf = [0.29912627861573327, 0.30886767304359375, 0.3088108573487326, 1.0657556075017878])
  end
  @testset "../examples/parameters_ec_mhd.toml" begin
    test_trixi_run("../examples/parameters_ec_mhd.toml",
            l2   = [0.03607862694368351, 0.04281395008247395, 0.04280207686965749, 0.025746770192645763, 0.1611518499414067, 0.017455917249117023, 0.017456981264942977, 0.02688321120361229, 0.00015024027267648003],
            linf = [0.23502083666166018, 0.3156846367743936, 0.31227895161037256, 0.2118146956106238, 0.9743049414302711, 0.09050624115026618, 0.09131633488909774, 0.15693063355520998, 0.0038394720095667593])
  end
  @testset "../examples/parameters_hyp_diff_harmonic_nonperiodic.toml" begin
    test_trixi_run("../examples/parameters_hyp_diff_harmonic_nonperiodic.toml",
            l2   = [8.618132353932638e-8, 5.619399844708813e-7, 5.619399845476024e-7],
            linf = [1.124861862326869e-6, 8.622436471483752e-6, 8.622436469707395e-6])
  end
  @testset "../examples/parameters_hyp_diff_llf.toml" begin
    test_trixi_run("../examples/parameters_hyp_diff_llf.toml",
            l2   = [5.944333465696809e-6, 3.744414248045495e-5, 3.744414248051497e-5],
            linf = [3.751627675097069e-5, 0.00019575235699065274, 0.00019575235699065274])
  end
  @testset "../examples/parameters_hyp_diff_nonperiodic.toml" begin
    test_trixi_run("../examples/parameters_hyp_diff_nonperiodic.toml",
            l2   = [8.449481442600338e-6, 2.978444363369848e-5, 5.590431119408737e-5],
            linf = [5.1550599501126726e-5, 0.00028988258669275583, 0.0003421368340976727])
  end
  @testset "../examples/parameters_hyp_diff_upwind.toml" begin
    test_trixi_run("../examples/parameters_hyp_diff_upwind.toml",
            l2   = [5.868147556488962e-6, 3.8051792732628014e-5, 3.8051792732620214e-5],
            linf = [3.70196549871471e-5, 0.0002072058411455302, 0.00020720584114464202])
  end
  @testset "../examples/parameters_mortar.toml" begin
    test_trixi_run("../examples/parameters_mortar.toml",
            l2   = [0.022356422238096973],
            linf = [0.5043638249003257])
  end
  @testset "../examples/parameters_mortar_vortex.toml" begin
    test_trixi_run("../examples/parameters_mortar_vortex.toml",
            l2   = [2.1202421511973067e-6, 2.7929028341308907e-5, 3.7593065314592924e-5, 8.813423453465327e-5],
            linf = [5.93205509794581e-5, 0.0007486675478352023, 0.0008175405566226424, 0.002212267888996422])
  end
  @testset "../examples/parameters_mortar_vortex_split.toml" begin
    test_trixi_run("../examples/parameters_mortar_vortex_split.toml",
            l2   = [2.1203040671963692e-6, 2.8053312800289536e-5, 3.761758762899687e-5, 8.840565162128428e-5],
            linf = [5.900575985384737e-5, 0.0007547236106317801, 0.000817616344069072, 0.0022090204216524967])
  end
  @testset "../examples/parameters_mortar_vortex_split_shockcapturing.toml" begin
    test_trixi_run("../examples/parameters_mortar_vortex_split_shockcapturing.toml",
            l2   = [2.1205855860697905e-6, 2.805356649496243e-5, 3.7617723084029226e-5, 8.841527980901164e-5],
            linf = [5.9005286894620035e-5, 0.0007547295163081724, 0.0008176139355887679, 0.0022089993378280326])
  end
  @testset "../examples/parameters_nonperiodic.toml" begin
    test_trixi_run("../examples/parameters_nonperiodic.toml",
            l2   = [2.3652137675654753e-6, 2.1386731303685556e-6, 2.138673130413185e-6, 6.009920290578574e-6],
            linf = [1.4080448659026246e-5, 1.7581818010814487e-5, 1.758181801525538e-5, 5.9568540361709665e-5])
  end
  @testset "../examples/parameters_source_terms.toml" begin
    test_trixi_run("../examples/parameters_source_terms.toml",
            l2   = [8.517783186497567e-7, 1.2350199409361865e-6, 1.2350199409828616e-6, 4.277884398786315e-6],
            linf = [8.357934254688004e-6, 1.0326389653148027e-5, 1.0326389654924384e-5, 4.4961900057316484e-5])
  end
  @testset "../examples/parameters_vortex.toml" begin
    test_trixi_run("../examples/parameters_vortex.toml",
            l2   = [3.6343138447409784e-6, 0.0032111379843728876, 0.0032111482778261658, 0.004545715889714643],
            linf = [7.901869034399045e-5, 0.030511158864742205, 0.030451936462313256, 0.04361908901631395])
  end
  @testset "../examples/parameters_vortex_split_shockcapturing.toml" begin
    test_trixi_run("../examples/parameters_vortex_split_shockcapturing.toml",
            l2   = [3.8034711509468997e-6, 5.561030973129845e-5, 5.563956603258559e-5, 0.00015706441614772137],
            linf = [8.493408680687597e-5, 0.0009610606296146518, 0.0009684675522437791, 0.003075812221315033])
  end
  @testset "../examples/parameters_weak_blast_wave_shockcapturing.toml" begin
    test_trixi_run("../examples/parameters_weak_blast_wave_shockcapturing.toml",
            l2   = [0.05365734539276933, 0.04683903386565478, 0.04684207891980008, 0.19632055541821553],
            linf = [0.18542234326379825, 0.24074440953554058, 0.23261143887822433, 0.687464986948263])
  end
  @test_nowarn Trixi.convtest("../examples/parameters.toml", 3)
  @test_skip   Trixi.run("../examples/parameters_blast_wave_shockcapturing_amr.toml") # errors for me
  @test_skip   Trixi.run("../examples/parameters_sedov_blast_wave_shockcapturing_amr.toml") # errors for me
end

# Only run extended tests if environment variable is set
if haskey(ENV, "TRIXI_TEST_EXTENDED") && lowercase(ENV["TRIXI_TEST_EXTENDED"]) in ("1", "on", "yes")
  @testset "Examples (long execution time)" begin
    @test_nowarn Trixi.run("../examples/parameters_blob.toml")
    @test_nowarn Trixi.run("../examples/parameters_blob_amr.toml")
    @test_nowarn Trixi.run("../examples/parameters_ec_performance_test.toml")
    @test_nowarn Trixi.run("../examples/parameters_khi.toml")
    @test_nowarn Trixi.run("../examples/parameters_ec_mortar.toml")
    @test_nowarn Trixi.run("../examples/parameters_khi_amr.toml")
    @test_nowarn Trixi.run("../examples/parameters_mhd_blast_wave.toml")
    @test_nowarn Trixi.run("../examples/parameters_orszag_tang.toml")
    @test_nowarn Trixi.run("../examples/parameters_rotor.toml")
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)
