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
  @testset "../examples/parameters.toml with N=1" begin
    test_trixi_run("../examples/parameters.toml",
            l2   = [0.05264106093598111],
            linf = [0.08754218386076518],
            N=1)
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
  @testset "../examples/parameters_mortar_vortex_split.toml with flux_central" begin
    test_trixi_run("../examples/parameters_mortar_vortex_split.toml",
            l2   = [2.1202421512026147e-6, 2.7929028341288412e-5, 3.759306531457842e-5, 8.813423453452753e-5],
            linf = [5.932055097812583e-5, 0.0007486675478027838, 0.0008175405566221983, 0.0022122678889928693],
            volume_flux = "flux_central")
  end
  @testset "../examples/parameters_mortar_vortex_split.toml with flux_kuya_etal" begin
    test_trixi_run("../examples/parameters_mortar_vortex_split.toml",
            l2   = [2.1179157511237394e-6, 2.805652962710273e-5, 3.7597878575795916e-5, 8.8405400202012e-5],
            linf = [5.913671873580828e-5, 0.0007547162965326759, 0.000816284733694328, 0.0022072012966312116],
            volume_flux = "flux_kuya_etal")
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
  @testset "../examples/parameters_khi_amr.toml with t_end=0.2" begin
    test_trixi_run("../examples/parameters_khi_amr.toml",
            l2   = [0.0016901662212296502, 0.005064145650514081, 0.004794017657493158, 0.0039877996168673525],
            linf = [0.027437774935491266, 0.02344577999610231, 0.016129408502293267, 0.018237901415986357],
            t_end = 0.2)
  end
  @testset "../examples/parameters_blob_amr.toml with t_end=0.12" begin
    test_trixi_run("../examples/parameters_blob_amr.toml",
            l2   = [0.20167272820008805, 1.1836133229138053, 0.10165112533393011, 5.237361125542303],
            linf = [14.085801194734044, 71.07468448364403, 7.366158173410174, 297.2413787328775],
            t_end = 0.12)
  end
  @testset "../examples/parameters_orszag_tang.toml with t_end=0.09" begin
    test_trixi_run("../examples/parameters_orszag_tang.toml",
            l2   = [0.22092878274261832, 0.26603025757015414, 0.31894245202569665, 0.0, 0.5232272243127627, 0.23339944630591253, 0.34777865111745, 0.0, 0.01475575698061458],
            linf = [1.3440153848990444, 0.6695453040530135, 0.8883866154398539, 0.0, 2.9725753256238696, 0.7228314117312579, 1.0384211822702207, 0.0, 0.14362546843006319],
            t_end = 0.09)
  end
  @testset "../examples/parameters_ec_mortar.toml with shock_capturing" begin
    test_trixi_run("../examples/parameters_ec_mortar.toml",
            l2   = [0.04816136246215661, 0.03713041026830962, 0.03713130328181323, 0.1777051166244772],
            linf = [0.3118606868100966, 0.34614370128998007, 0.3460122144359348, 1.1085840270633454],
            volume_integral_type = "shock_capturing")
  end
  @testset "../examples/parameters.toml with restart and t_end=2" begin
    Trixi.run("../examples/parameters.toml")
    test_trixi_run("../examples/parameters.toml",
            l2   = [1.2148032444677485e-5],
            linf = [6.495644794757283e-5],
            t_end = 2,
            restart = true, restart_filename = "out/restart_000040.h5")
  end
  @test_nowarn Trixi.convtest("../examples/parameters.toml", 3)
  @test_skip   Trixi.run("../examples/parameters_blast_wave_shockcapturing_amr.toml") # errors for me
  @test_skip   Trixi.run("../examples/parameters_sedov_blast_wave_shockcapturing_amr.toml") # errors for me
end

# Coverage test for all initial conditions
@testset "Tests for initial conditions" begin
  # Linear scalar advection
  @testset "../examples/parameters.toml with initial_conditions_sin_sin" begin
    test_trixi_run("../examples/parameters.toml",
            l2   = [0.0001424424804667062],
            linf = [0.0007260692243250544],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_sin_sin")
  end
  @testset "../examples/parameters.toml with initial_conditions_constant" begin
    test_trixi_run("../examples/parameters.toml",
            l2   = [6.120436421866528e-16],
            linf = [1.3322676295501878e-15],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_constant")
  end
  @testset "../examples/parameters.toml with initial_conditions_linear_x_y" begin
    test_trixi_run("../examples/parameters.toml",
            l2   = [0.28716994072507784],
            linf = [5.0513408],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_linear_x_y")
  end
  @testset "../examples/parameters.toml with initial_conditions_linear_x" begin
    test_trixi_run("../examples/parameters.toml",
            l2   = [0.201514980654101],
            linf = [2.525670400000001],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_linear_x")
  end
  @testset "../examples/parameters.toml with initial_conditions_linear_y" begin
    test_trixi_run("../examples/parameters.toml",
            l2   = [0.20151498065410095],
            linf = [2.525670400000001],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_linear_y")
  end
  # Compressible Euler
  @testset "../examples/parameters_vortex.toml one step with initial_conditions_density_pulse" begin
    test_trixi_run("../examples/parameters_vortex.toml",
            l2   = [0.003201074851451383, 0.0032010748514513724, 0.0032010748514513716, 0.0032010748514513794],
            linf = [0.043716393835876444, 0.043716393835876444, 0.043716393835876, 0.04371639383587578],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_density_pulse")
  end
  @testset "../examples/parameters_vortex.toml one step with initial_conditions_pressure_pulse" begin
    test_trixi_run("../examples/parameters_vortex.toml",
            l2   = [0.00018950189533270512, 0.0020542290689775757, 0.002054229068977579, 0.01013381064979542],
            linf = [0.004763284475434837, 0.028439617580275578, 0.028439617580275467, 0.13640572175447918],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_pressure_pulse")
  end
  @testset "../examples/parameters_vortex.toml one step with initial_conditions_density_pressure_pulse" begin
    test_trixi_run("../examples/parameters_vortex.toml",
            l2   = [0.0031880440066425803, 0.0050397619349217574, 0.005039761934921767, 0.014340770024960708],
            linf = [0.04279723800834989, 0.06783565847184869, 0.06783565847184914, 0.19291274039254347],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_density_pressure_pulse")
  end
  @testset "../examples/parameters_vortex.toml one step with initial_conditions_constant" begin
    test_trixi_run("../examples/parameters_vortex.toml",
            l2   = [2.359732835648237e-16, 1.088770274131804e-16, 1.1814939065033234e-16, 1.980283448445849e-15],
            linf = [4.440892098500626e-16, 2.914335439641036e-16, 4.718447854656915e-16, 3.552713678800501e-15],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_constant")
  end
  @testset "../examples/parameters_sedov_blast_wave_shockcapturing_amr.toml one step" begin
    test_trixi_run("../examples/parameters_sedov_blast_wave_shockcapturing_amr.toml",
            l2   = [0.002911075352335366, 0.01249799423742342, 0.01249799423742343, 0.11130739933709777],
            linf = [0.15341072072011042, 0.763322686048535, 0.7633226860485351, 5.184635785270958],
            n_steps_max = 1)
  end
  @testset "../examples/parameters_sedov_blast_wave_shockcapturing_amr.toml one step with initial_conditions_medium_sedov_blast_wave" begin
    test_trixi_run("../examples/parameters_sedov_blast_wave_shockcapturing_amr.toml",
            l2   = [0.0029095199084281176, 0.012491250999308508, 0.012491250999308522, 0.11126623649275227],
            linf = [0.15334906997459008, 0.7629367729245761, 0.7629367729245761, 5.18264418672338],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_medium_sedov_blast_wave")
  end

  # GLM-MHD
  @testset "../examples/parameters_alfven_wave.toml one step with initial_conditions_constant" begin
    test_trixi_run("../examples/parameters_alfven_wave.toml",
            l2   = [2.09587921e-16, 2.00505223e-16, 4.67253468e-16, 1.07374002e-16, 4.12845635e-15, 4.50940304e-16, 2.82296601e-16, 1.09431793e-16, 1.76565492e-16],
            linf = [3.33066907e-16, 7.49400542e-16, 1.77635684e-15, 1.66533454e-16, 1.42108547e-14, 8.88178420e-16, 6.66133815e-16, 1.66533454e-16, 7.41458237e-16],
            n_steps_max = 1,
            initial_conditions = "initial_conditions_constant")
  end
  @testset "../examples/parameters_rotor.toml one step" begin
    test_trixi_run("../examples/parameters_rotor.toml",
            l2   = [0.019427744975023344, 0.04740447527885731, 0.043330457224485704, 0.0, 0.07742181652703439, 0.0071460140577900575, 0.00984439127521705, 0.0, 0.0021335541010966083],
            linf = [0.5279494653654151, 1.9501858268442724, 1.9517180876506082, 0.0, 2.9931160992455936, 0.389052695580107, 0.3907824507405411, 0.0, 0.1822683069110969],
            n_steps_max = 1)
  end
  @test_skip Trixi.run("../examples/parameters_mhd_blast_wave.toml", n_steps_max=1)
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
