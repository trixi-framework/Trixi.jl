module TestExamples3D  # TODO: Taal remove

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

# Run basic tests
@testset "Examples 3D" begin
  @testset "taal-confirmed parameters_advection_basic.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_advection_basic.toml"),
            l2   = [0.00015975754755823664],
            linf = [0.001503873297666436])
  end
  @testset "taal-confirmed parameters_advection_basic.toml with restart and t_end=2" begin
    Trixi.run(joinpath(EXAMPLES_DIR, "parameters_advection_basic.toml"))
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_advection_basic.toml"),
            l2   = [0.00017800012466353434],
            linf = [0.001452075263740804],
            t_end = 2, restart = true, restart_filename = "out/restart_000040.h5")
  end
  @testset "taal-confirmed parameters_advection_basic.toml with initial_condition_sin" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_advection_basic.toml"),
            l2   = [0.002727292086517533],
            linf = [0.024833049753677727],
            initial_condition=Trixi.initial_condition_sin)
  end
  @testset "taal-confirmed parameters_advection_basic.toml with initial_condition_constant" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_advection_basic.toml"),
            l2   = [9.770171014620371e-16],
            linf = [2.4424906541753444e-15],
            initial_condition=initial_condition_constant)
  end
  @testset "taal-confirmed parameters_advection_basic.toml with initial_condition_linear_z and periodicity=false" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_advection_basic.toml"),
            l2   = [6.607840408143593e-16],
            linf = [5.773159728050814e-15],
            initial_condition=Trixi.initial_condition_linear_z,
            boundary_conditions=Trixi.boundary_condition_linear_z, periodicity=false)
  end
  @testset "taal-confirmed parameters_euler_source_terms.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_source_terms.toml"),
            l2   = [0.010323099666828388, 0.00972876713766357, 0.00972876713766343, 0.009728767137663324, 0.015080409341036285],
            linf = [0.034894880154510144, 0.03383545920056008, 0.033835459200560525, 0.03383545920054587, 0.06785780622711979])
  end
  @testset "taal-confirmed parameters_euler_source_terms.toml with split_form" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_source_terms.toml"),
            l2   = [0.010323099666828388, 0.00972876713766357, 0.00972876713766343, 0.009728767137663324, 0.015080409341036285],
            linf = [0.034894880154510144, 0.03383545920056008, 0.033835459200560525, 0.03383545920054587, 0.06785780622711979],
            volume_integral_type = "split_form")
  end
  @testset "taal-confirmed parameters_euler_eoc_test.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_eoc_test.toml"),
            l2   = [0.000363723832448333, 0.00039555684672049366, 0.0003955568467203738, 0.00039555684672064724, 0.0007811604790242773],
            linf = [0.002400072140187337, 0.0029635489437536133, 0.0029635489437540574, 0.0029635489437565, 0.007191455734479657])
  end
  @testset "taal-confirmed parameters_eulergravity_eoc_test.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_eulergravity_eoc_test.toml"),
            l2   = [0.00042767972112699913, 0.00047204316046796835, 0.00047204316046784795, 0.0004720431604680035, 0.0010987015429634586, 0.00012296598036447797, 0.0005745341792812197, 0.0005745341792812442, 0.0005745341792812238],
            linf = [0.0034966337661186397, 0.0037643976198782347, 0.003764397619878901, 0.0037643976198780127, 0.008370354378078648, 0.0010129211321238465, 0.0024406779290754455, 0.002440677929075438, 0.0024406779290755756],
            resid_tol = 1.0e-4, t_end = 0.2)
  end
  @testset "taal-confirmed parameters_advection_mortar.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_advection_mortar.toml"),
            l2   = [0.0018461483161353273],
            linf = [0.017728496545256434])
  end
  @testset "taal-confirmed parameters_euler_mortar.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_mortar.toml"),
            l2   = [0.0019011097431965655, 0.0018289464087588392, 0.0018289464087585998, 0.0018289464087588862, 0.003354766311541738],
            linf = [0.011918594206950184, 0.011808582644224241, 0.011808582644249999, 0.011808582644239785, 0.02464803617735356])
  end
  @testset "taal-confirmed parameters_advection_amr.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_advection_amr.toml"),
            l2   = [9.773858425669403e-6],
            linf = [0.0005853874124926092])
  end
  @testset "taal-confirmed parameters_euler_amr.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_amr.toml"),
            l2   = [0.00382819196730758, 0.0038281919673075725, 0.0038281919673075746, 0.0038281919673075738, 0.0057422879509614905],
            linf = [0.07390560349428554, 0.07390560349428577, 0.07390560349428621, 0.07390560349428643, 0.11085840524143098],
            t_end=0.1)
  end
  @testset "taal-confirmed parameters_euler_blob_amr.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_blob_amr.toml"),
            l2   = [0.04641288111176061, 0.25587674204071276, 0.03358077695134774, 0.033580776951347646, 1.0552376897521447],
            linf = [9.773023537783775, 48.58018285242446, 9.56936427666669, 9.56936427666669, 170.97251488955624],
            t_end=0.2)
  end
  @testset "taal-confirmed parameters_euler_ec.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_ec.toml"),
            l2   = [0.025101741317688664, 0.01655620530022176, 0.016556205300221737, 0.016549388264402515, 0.09075092792976944],
            linf = [0.43498932208478724, 0.2821813924028202, 0.28218139240282025, 0.2838043627560838, 1.5002293438086647])
  end
  @testset "taal-confirmed parameters_euler_ec.toml with initial_condition=initial_condition_constant" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_ec.toml"),
            l2   = [5.717218008425079e-16, 6.088971423170968e-16, 6.23130776282275e-16, 7.29884557381127e-16, 5.167198077601542e-15],
            linf = [3.885780586188048e-15, 4.454769886308441e-15, 3.219646771412954e-15, 4.884981308350689e-15, 4.440892098500626e-14],
            initial_condition=initial_condition_constant)
  end
  @testset "taal-confirmed parameters_euler_ec.toml with flux_chandrashekar" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_ec.toml"),
            l2   = [0.025105743648126774, 0.016571417754430256, 0.01657141775443023, 0.016565202090289916, 0.09077232065771225],
            linf = [0.4349225166034201, 0.27945714200874, 0.2794571420087401, 0.28021366413271664, 1.5240679700745954],
            surface_flux=flux_chandrashekar, volume_flux=flux_chandrashekar)
  end
  @testset "taal-confirmed parameters_euler_ec.toml with flux_kennedy_gruber" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_ec.toml"),
            l2   = [0.025120431810845507, 0.016599310737401483, 0.01659931073740148, 0.016592567464138185, 0.090856457771812],
            linf = [0.43120500632996794, 0.28419288751363336, 0.2841928875136334, 0.28583515705222146, 1.515485025725378],
            surface_flux=flux_kennedy_gruber, volume_flux=flux_kennedy_gruber)
  end
  @testset "taal-confirmed parameters_euler_ec.toml with flux_shima_etal" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_ec.toml"),
            l2   = [0.025099944530993942, 0.016561611274319134, 0.016561611274319127, 0.01655478190136039, 0.09076538812894279],
            linf = [0.43472962954165273, 0.2824065323711477, 0.2824065323711474, 0.28409419760015847, 1.4995295774522692],
            surface_flux=flux_shima_etal, volume_flux=flux_shima_etal)
  end
  @testset "taal-confirmed parameters_euler_density_pulse.toml with source_terms_harmonic" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_density_pulse.toml"),
            l2   = [0.05719652660597408, 0.0571965266059741, 0.05719652660597407, 0.05719652660597409, 0.08579478990896279],
            linf = [0.27375961853433606, 0.27375961853433517, 0.27375961853433384, 0.2737596185343343, 0.4106394278015033],
            source_terms=Trixi.source_terms_harmonic,
            extra_analysis_quantities=["l2_error_primitive", "linf_error_primitive"])
  end
  @testset "taal-confirmed parameters_euler_taylor_green_vortex.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_taylor_green_vortex.toml"),
            l2   = [0.0003494971047256544, 0.03133386380969968, 0.031333863809699644, 0.04378595081016185, 0.015796569210801217],
            linf = [0.0013934701399120897, 0.07284947983025436, 0.07284947983025408, 0.12803234075782724, 0.07624639122292365],
            t_end = 0.5)
  end
  @testset "taal-confirmed parameters_euler_shockcapturing.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_shockcapturing.toml"),
            l2   = [0.025558219399128387, 0.01612806446620796, 0.016128064466207948, 0.016120400619198158, 0.09208276987000782],
            linf = [0.3950327737713353, 0.26324766244272796, 0.2632476624427279, 0.2634129727753079, 1.371321006006725])
  end
  @testset "taal-confirmed parameters_euler_shockcapturing.toml with initial_condition_sedov_blast_wave" begin
    # OBS! This setup does not run longer but crashes (also the parameters do not make sense) -> only for testing the IC!
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_shockcapturing.toml"),
            l2   = [0.03627060784392582, 0.05178777376859809, 0.05178777376859804, 0.05178777376859802, 0.23043996953698023],
            linf = [0.9307998233177583, 1.4326649193439467, 1.4326649193439467, 1.4326649193439467, 12.80585041235138],
            initial_condition=initial_condition_sedov_blast_wave, cfl=0.25, shock_alpha_max=1.0, t_end=0.1)
  end
  @testset "taal-confirmed parameters_euler_shockcapturing.toml with initial_condition_sedov_self_gravity" begin
    # OBS! This setup does not run longer but crashes (also the parameters do not make sense) -> only for testing the IC!
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_shockcapturing.toml"),
            l2   = [0.04846527000320781, 0.051787773760055514, 0.051787773760055486, 0.05178777376005548, 0.23043996953467236],
            linf = [0.9307979866990295, 1.4326649193456429, 1.4326649193456429, 1.4326649193456429, 12.805850412386896],
            initial_condition=initial_condition_sedov_self_gravity, cfl=0.25, shock_alpha_max=1.0, t_end=0.1)
  end
  @testset "taal-confirmed parameters_euler_shockcapturing.toml with amr_indicator=`sedov_self_gravity`" begin
    # OBS! This setup does not make much practical sense. It is only added to exercise the
    # `sedov_self_gravity` AMR indicator, which in its original configuration is too expensive for
    # CI testing
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_shockcapturing.toml"),
            l2   = [0.022890627324485553, 0.013353127563710173, 0.013353127563710156, 0.013352146925243637, 0.08348119429398775],
            linf = [0.3769085161148348, 0.380358363190641, 0.3803583631906434, 0.380366775575835, 1.3162027903728162],
            n_steps_max=10, amr_interval=1, amr_indicator="sedov_self_gravity", max_refinement_level=4)
  end
  @testset "taal-confirmed parameters_hyp_diff_llf.toml with initial_refinement_level=2" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_hyp_diff_llf.toml"),
            l2   = [0.0015303292770225546, 0.011314166522881952, 0.011314166522881981, 0.011314166522881947],
            linf = [0.022634590339093097, 0.10150613595329361, 0.10150613595329361, 0.10150613595329361],
            initial_refinement_level=2)
  end
  @testset "taal-confirmed parameters_hyp_diff_llf.toml with initial_refinement_level=2, surface_flux=flux_upwind)" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_hyp_diff_llf.toml"),
            l2   = [0.0015377708559180534, 0.011376842329542572, 0.011376842329542624, 0.0113768423295426],
            linf = [0.02271542063004106, 0.10191067906109286, 0.10191067906109552, 0.10191067906109286],
            initial_refinement_level=2, surface_flux=flux_upwind)
  end
  @testset "taal-confirmed parameters_hyp_diff_nonperiodic.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_hyp_diff_nonperiodic.toml"),
            l2   = [0.00022868324220593294, 0.0007974310370259415, 0.0015035143239197598, 0.0015035143239198418],
            linf = [0.0016329580288680923, 0.0029870270738030775, 0.009177053066089513, 0.009177053066084184])
  end
  @testset "taal-confirmed parameters_mhd_ec.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_mhd_ec.toml"),
            l2   = [0.01921453037426997, 0.01924853398980921, 0.01924853398980923, 0.019247118340533328, 0.08310482412935676, 0.010362656540935251, 0.010362656540935237, 0.010364587080559528, 0.00020760700572485828],
            linf = [0.2645851360519166, 0.33611482816103344, 0.33611482816103466, 0.36952265576762666, 1.230825809630423, 0.09818527443798974, 0.09818527443798908, 0.10507242371450054, 0.008456471524217968])
  end
  @testset "taal-confirmed parameters_mhd_ec.toml with initial_condition=initial_condition_constant" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_mhd_ec.toml"),
            l2   = [4.850506049646793e-16, 2.4804155700127237e-15, 3.579471462379534e-15, 2.7395862184339726e-15, 2.4916602560342516e-14, 1.669368799061149e-15, 1.4052897861706032e-15, 1.0685989093080367e-15, 1.1611070325375158e-15],
            linf = [3.552713678800501e-15, 1.4710455076283324e-14, 2.3814283878209608e-14, 2.6423307986078726e-14, 1.6342482922482304e-13, 1.1546319456101628e-14, 1.0880185641326534e-14, 1.4099832412739488e-14, 1.1483287543575534e-14],
            atol = 1000*eps(),
            initial_condition=initial_condition_constant)
  end
  @testset "taal-confirmed parameters_mhd_alfven_wave.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_mhd_alfven_wave.toml"),
            l2   = [0.0038729054515012624, 0.00903693761037057, 0.0041729297273898815, 0.01160504558506348, 0.006241548790045999, 0.009227641613254402, 0.0034580608435846143, 0.011684993365513006, 0.0022068452165023645],
            linf = [0.012628629484152443, 0.03265276295369954, 0.012907838374176334, 0.044746702024108326, 0.02796611265824822, 0.03453054781110626, 0.010261557301859958, 0.044762592434299864, 0.010012319622784436])
  end
  @testset "taal-confirmed parameters_mhd_alfven_wave_mortar.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_mhd_alfven_wave_mortar.toml"),
            l2   = [0.0021484102061835623, 0.006826504155492453, 0.0030653111370061784, 0.008735898256361025, 0.0051601878379492335, 0.007157480202233399, 0.0028291977973972948, 0.008815052614117018, 0.0022321821323698257],
            linf = [0.012956870409227328, 0.05529249146399706, 0.020854504834048836, 0.05898012498637771, 0.03162799656904003, 0.05512773554440975, 0.017941374395225362, 0.060061114374191496, 0.013036070296136178],
            t_end = 0.25)
  end
  # 3D Orszag-Tang included to exercise all terms in the HLL flux
  @testset "taal-confirmed parameters_mhd_orszag_tang.toml" begin
    test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_mhd_orszag_tang.toml"),
            l2   = [0.0043911605751115424, 0.04144735653371165, 0.04150129965650717, 0.04150353600000829, 0.036931197750736805, 0.021125598820694595, 0.032956068087418154, 0.03296235602392588, 6.318083915607208e-6],
            linf = [0.01789383976134809, 0.08496187610572214, 0.08909116075943745, 0.08505952838326755, 0.10443373959204932, 0.05387852204182135, 0.08812990990777562, 0.07804874749131957, 8.138512446081734e-5],
            t_end = 0.06)
  end
  # too expensive for CI
  # @testset "taal-confirmed parameters_euler_sedov_blast_wave.toml with n_steps_max = 2" begin
  #   test_trixi_run(joinpath(EXAMPLES_DIR, "parameters_euler_sedov_blast_wave.toml"),
  #           l2   = [0.00015213881280510253, 0.001481110249423103, 0.0014811102494231387, 0.001481110249423187, 0.002940437008367858],
  #           linf = [0.03254534843490764, 0.38932044051654113, 0.38932044051654097, 0.38932044051654097, 1.050399588579145],
  #           n_steps_max = 2)
  # end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end #module
