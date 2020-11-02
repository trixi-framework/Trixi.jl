module TestExamples2D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "2D" begin

# Run basic tests
@testset "Examples 2D" begin
  @testset "taal-confirmed elixir_advection_basic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.144681765639205e-6],
      linf = [6.437440532547356e-5])
  end

  @testset "taal-confirmed elixir_advection_mortar.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
      l2   = [0.022356422238096973],
      linf = [0.5043638249003257])
  end

  @testset "taal-confirmed elixir_advection_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [0.010844189678803203],
      linf = [0.0491178481591637])
  end

  @testset "taal-check-me elixir_advection_restart.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [1.2148032444677485e-5],
      linf = [6.495644794757283e-5])
  end


  @testset "taal-check-me elixir_hyp_diff_llf.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_llf.jl"),
      l2   = [0.0001568775181745306, 0.001025986772217103, 0.0010259867722170538],
      linf = [0.0011986956378152724, 0.006423873516111733, 0.006423873516110845])
  end

  @testset "taal-confirmed elixir_hyp_diff_harmonic_nonperiodic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_harmonic_nonperiodic.jl"),
      l2   = [8.618132353932638e-8, 5.619399844708813e-7, 5.619399845476024e-7],
      linf = [1.124861862326869e-6, 8.622436471483752e-6, 8.622436469707395e-6])
  end

  @testset "taal-confirmed elixir_hyp_diff_nonperiodic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_nonperiodic.jl"),
      l2   = [8.523077654037775e-6, 2.877932365308637e-5, 5.454942769137812e-5],
      linf = [5.484978959957587e-5, 0.00014544895979200218, 0.000324491268921534])
  end

  @testset "taal-confirmed elixir_hyp_diff_upwind.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_upwind.jl"),
      l2   = [5.868147556488962e-6, 3.8051792732628014e-5, 3.8051792732620214e-5],
      linf = [3.70196549871471e-5, 0.0002072058411455302, 0.00020720584114464202])
  end


  @testset "taal-confirmed elixir_euler_source_terms.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [8.517783186497567e-7, 1.2350199409361865e-6, 1.2350199409828616e-6, 4.277884398786315e-6],
      linf = [8.357934254688004e-6, 1.0326389653148027e-5, 1.0326389654924384e-5, 4.4961900057316484e-5])
  end

  @testset "taal-confirmed elixir_euler_density_wave.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.001060077845747576, 0.00010600778457107525, 0.00021201556914875742, 2.6501946139091318e-5],
      linf = [0.0065356386867677085, 0.0006535638688170142, 0.0013071277374487877, 0.0001633909674296774],
      tspan = (0.0, 0.5))
  end

  @testset "taal-confirmed elixir_euler_nonperiodic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic.jl"),
      l2   = [2.3652137675654753e-6, 2.1386731303685556e-6, 2.138673130413185e-6, 6.009920290578574e-6],
      linf = [1.4080448659026246e-5, 1.7581818010814487e-5, 1.758181801525538e-5, 5.9568540361709665e-5])
  end

  @testset "taal-confirmed elixir_euler_ec.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.06159341742582756, 0.05012484425381723, 0.05013298724507752, 0.22537740506116724],
      linf = [0.29912627861573327, 0.30886767304359375, 0.3088108573487326, 1.0657556075017878])
  end

  @testset "taal-confirmed elixir_euler_blast_wave_shockcapturing.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_shockcapturing.jl"),
      l2   = [0.13910202327088322, 0.11538722576277083, 0.1153873048510009, 0.3387876385945495],
      linf = [1.454418325889352, 1.3236875559310013, 1.323687555933169, 1.8225476335086368],
      maxiters=30)
  end

  @testset "taal-confirmed elixir_euler_weak_blast_wave_shockcapturing.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weak_blast_wave_shockcapturing.jl"),
      l2   = [0.05365734539276933, 0.04683903386565478, 0.04684207891980008, 0.19632055541821553],
      linf = [0.18542234326379825, 0.24074440953554058, 0.23261143887822433, 0.687464986948263])
  end
  @testset "taal-confirmed elixir_euler_blast_wave_shockcapturing_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_shockcapturing_amr.jl"),
    l2   = [0.6776486969229696, 0.2813026529898539, 0.2813025645101231, 0.7174702524881597],
    linf = [2.8939055423031546, 1.7997630098946877, 1.7997118659969253, 3.0341223482585686],
    tspan = (0.0, 1.0))
  end

  @testset "taal-confirmed elixir_euler_sedov_blast_wave_shockcapturing_amr.jl with tend = 1.0" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_shockcapturing_amr.jl"),
    l2   = [0.4820048896322639, 0.16556563003698888, 0.16556563003698901, 0.643610807739157],
    linf = [2.485752556439829, 1.2870638985941658, 1.2870638985941667, 6.474544663221404],
    tspan = (0.0, 1.0))
  end

  @testset "taal-confirmed elixir_euler_sedov_blast_wave_shockcapturing_amr.jl one step" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_shockcapturing_amr.jl"),
      l2   = [0.0021037031798961936, 0.010667428589443041, 0.010667428589443027, 0.11041565217737695],
      linf = [0.11754829172684966, 0.7227194329885249, 0.7227194329885249, 5.42708544137305],
      maxiters=1)
  end

  @testset "taal-confirmed parameters_euler_sedov_blast_wave_shockcapturing_amr.toml one step with initial_condition_medium_sedov_blast_wave" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_shockcapturing_amr.jl"),
      l2   = [0.002102553227287478, 0.01066154856802227, 0.010661548568022277, 0.11037470219676422],
      linf = [0.11749257043751615, 0.7223475657303381, 0.7223475657303381, 5.425015419074852],
      maxiters=1, initial_condition=initial_condition_medium_sedov_blast_wave)
  end

  @testset "taal-check-me elixir_euler_blob_shockcapturing_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_shockcapturing_amr.jl"),
      l2   = [0.2012143467980036, 1.1813241716700988, 0.10144725208346557, 5.230607564921326],
      linf = [14.111578610092542, 71.21944410118338, 7.304666476530256, 291.9385076318331],
      tspan = (0.0, 0.12))
  end

  @testset "taal-check-me elixir_euler_khi_shockcapturing.jl" begin
    if Threads.nthreads() == 1
      # This example uses random numbers to generate the initial condition.
      # Hence, we can only check "errors" if everything is made reproducible.
      # However, that's not enough to ensure reproducibility since the stream
      # of random numbers is not guaranteed to be the same across different
      # minor versions of Julia.
      # See https://github.com/trixi-framework/Trixi.jl/issues/232#issuecomment-709738400
      test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi_shockcapturing.jl"),
        l2   = [0.0020460050625351277, 0.0028624298590723372, 0.001971035381754319, 0.004814883331768111],
        linf = [0.02437585564403255, 0.018033033465721604, 0.00993916546672498, 0.02097263472404709],
        tspan = (0.0, 0.2))
    else
      test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi_shockcapturing.jl"),
        tspan = (0.0, 0.2))
    end
  end

  @testset "taal-check-me elixir_euler_khi_shockcapturing_amr.jl" begin
    if Threads.nthreads() == 1
      # This example uses random numbers to generate the initial condition.
      # Hence, we can only check "errors" if everything is made reproducible.
      # However, that's not enough to ensure reproducibility since the stream
      # of random numbers is not guaranteed to be the same across different
      # minor versions of Julia.
      # See https://github.com/trixi-framework/Trixi.jl/issues/232#issuecomment-709738400
      test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi_shockcapturing_amr.jl"),
        l2   = [0.001617236176233394, 0.0023394729603446697, 0.001296199247911843, 0.0033150160736185323],
        linf = [0.019002843896656074, 0.017242107049387223, 0.008179888370650977, 0.016885672229959958],
        tspan = (0.0, 0.2))
    else
      test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi_shockcapturing_amr.jl"),
        tspan = (0.0, 0.2))
    end
  end
  
  @testset "taal-check-me elixir_euler_vortex.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
    l2   = [3.6342636871275523e-6, 0.0032111366825032443, 0.0032111479254594345, 0.004545714785045611],
    linf = [7.903587114788113e-5, 0.030561314311228993, 0.030502600162385596, 0.042876297246817074])
  end
  
  @testset "taal-check-me elixir_euler_vortex_mortar.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
    l2   = [2.120307461394424e-6, 2.7929229084570266e-5, 3.759342242369596e-5, 8.813646673773311e-5],
    linf = [5.9320459189771135e-5, 0.0007491265403041236, 0.0008165690047976515, 0.0022122638048145404])
  end
  
  @testset "taal-check-me elixir_euler_vortex_mortar_split.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.1203693476896995e-6, 2.8053512416422296e-5, 3.76179445622429e-5, 8.840787521479401e-5],
    linf = [5.9005667252809424e-5, 0.0007554116730550398, 0.00081660478740464, 0.002209016304192346])
  end
  
  @testset "taal-check-me elixir_euler_vortex_mortar_split.jl with flux_central" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.120307461409829e-6, 2.7929229084583212e-5, 3.759342242369501e-5, 8.813646673812448e-5],
    linf = [5.932045918888296e-5, 0.0007491265403021252, 0.0008165690047987617, 0.002212263804818093],
    volume_flux = flux_central)
  end
  
  @testset "taal-check-me elixir_euler_vortex_mortar_split.jl with flux_shima_etal" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.120103291509122e-6, 2.805652562691104e-5, 3.759500428816484e-5, 8.841374592860891e-5],
    linf = [5.934103184424e-5, 0.0007552316820342853, 0.0008152449048961508, 0.002206987374638203],
    volume_flux = flux_shima_etal)
  end
  
  @testset "taal-check-me elixir_euler_vortex_mortar_split.jl with flux_ranocha" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.1201032806889955e-6, 2.8056528074361895e-5, 3.759500957406334e-5, 8.841379428954133e-5],
    linf = [5.934027760512439e-5, 0.0007552314317718078, 0.0008152450117491217, 0.0022069976113101575],
    volume_flux = flux_ranocha)
  end
  
  @testset "taal-check-me elixir_euler_vortex_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
    l2   = [3.80342739421474e-6, 5.561118953968859e-5, 5.564042529709319e-5, 0.0001570628548096201],
    linf = [8.491382365727329e-5, 0.0009602965158113097, 0.0009669978616948516, 0.0030750353269972663])
  end
  
  @testset "taal-check-me elixir_euler_vortex_mortar_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_shockcapturing.jl"),
    l2   = [2.1203693476896995e-6, 2.8053512416422296e-5, 3.76179445622429e-5, 8.840787521479401e-5],
    linf = [5.9005667252809424e-5, 0.0007554116730550398, 0.00081660478740464, 0.002209016304192346])
  end
  
  @testset "taal-check-me elixir_euler_vortex_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
    l2   = [2.077084130934081e-6, 0.0032815991956917493, 0.0032807020145523757, 0.004646298951577697],
    linf = [4.435791998502747e-5, 0.03176757178286449, 0.031797053799604846, 0.045615287239808566])
  end

  @testset "taal-confirmed elixir_mhd_alfven_wave.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.00011134513490658689, 5.880188909157728e-6, 5.880188909159547e-6, 8.432880997656317e-6, 1.2942387343501909e-6, 1.2238820298971968e-6, 1.2238820298896402e-6, 1.830621754702352e-6, 8.071881352562945e-7],
      linf = [0.00025632790161078667, 1.6379021163651086e-5, 1.637902116437273e-5, 2.58759953227633e-5, 5.327732286231068e-6, 8.118520269495555e-6, 8.118520269606577e-6, 1.2107354757678879e-5, 4.165806320060789e-6])
  end

  @testset "taal-confirmed elixir_mhd_alfven_wave_mortar.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
      l2   = [4.608223422391918e-6, 1.6891556053250136e-6, 1.6202140809698534e-6, 1.6994400213969954e-6, 1.4856807283318347e-6, 1.387768347373047e-6, 1.3411738859512443e-6, 1.7155298750074954e-6, 9.799762075600064e-7],
      linf = [3.52219535260101e-5, 1.534468550207224e-5, 1.426263439847919e-5, 1.4421456102198249e-5, 7.743399239257265e-6, 1.019242699840106e-5, 9.862935257842764e-6, 1.6018118328936515e-5, 5.563695788849475e-6],
      tspan = (0.0, 1.0))
  end

  @testset "taal-confirmed elixir_mhd_ec.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.03607862694368351, 0.04281395008247395, 0.04280207686965749, 0.025746770192645763, 0.1611518499414067, 0.017455917249117023, 0.017456981264942977, 0.02688321120361229, 0.00015024027267648003],
      linf = [0.23502083666166018, 0.3156846367743936, 0.31227895161037256, 0.2118146956106238, 0.9743049414302711, 0.09050624115026618, 0.09131633488909774, 0.15693063355520998, 0.0038394720095667593])
  end

  @testset "taal-check-me elixir_mhd_orszag_tang_shockcapturing_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang_shockcapturing_amr.jl"),
    l2   = [0.21894721281911703, 0.26463302881957645, 0.31507117273918805, 0.0, 0.5152448296476039, 0.23023274798808147, 0.3441658797437742, 0.0, 0.0026733194007546126],
    linf = [1.2352286192592534, 0.6678377088690369, 0.8739431671403393, 0.0, 2.740788100988533, 0.6552251870441527, 0.9546253266155187, 0.0, 0.03816123862195953],
    tspan = (0.0, 0.09))
  end

  @testset "taal-check-me elixir_mhd_orszag_tang_shockcapturing_amr.jl with flux_hll" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang_shockcapturing_amr.jl"),
    l2   = [0.10811766489910432, 0.20202956511451342, 0.22988158838731435, 0.0, 0.29953446216629687, 0.1570994904887061, 0.24308871328334844, 0.0, 0.011100323402918071],
    linf = [0.5520018702830969, 0.5101514485370506, 0.6565173233469559, 0.0, 0.9528527119850311, 0.3990329190790233, 0.6737022346309564, 0.0, 0.18244193667531056],
    tspan = (0.0, 0.06), surface_flux = flux_hll)
  end

  @testset "taal-check-me elixir_mhd_rotor_shockcapturing_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor_shockcapturing_amr.jl"),
    l2   = [1.2635449181120562, 1.8356372101225815, 1.7037178920138905, 0.0, 2.3126474248436755, 0.21626214510814928, 0.23683073618598693, 0.0, 0.002132844459180628],
    linf = [10.353812749882609, 14.287005221052532, 15.749922601372482, 0.0, 17.089103075830185, 1.342006287193983, 1.4341241435029897, 0.0, 0.053488038358224646],
    tspan = (0.0, 0.05))
  end

  @testset "taal-check-me elixir_mhd_blast_wave_shockcapturing_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave_shockcapturing_amr.jl"),
    l2   = [0.2101138028554417, 4.4379574949560014, 2.6239651859752238, 0.0, 359.15092246795564, 2.458555512327778, 1.4961525378625697, 0.0, 0.01346996306689436],
    linf = [2.4484577379812915, 63.229017006957584, 15.321798382742966, 0.0, 2257.8231751993367, 13.692356305778407, 10.026947993726841, 0.0, 0.2839557716528234],
    tspan = (0.0, 0.003))
  end

  @testset "taal-check-me elixir_euler_gravity_jeans_instability.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_gravity_jeans_instability.jl"),
    l2   = [10733.633835334986, 13356.780418347276, 1.6387671498796526e-6, 26834.07694603585],
    linf = [15194.296497141942, 18881.481420845837, 8.91191153038996e-6, 37972.99718572572],
    tspan = (0.0, 0.1))
  end

  @testset "taal-check-me elixir_euler_gravity_sedov_blast_wave_shockcapturing_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_gravity_sedov_blast_wave_shockcapturing_amr.jl"),
    l2   = [0.04630745182870653, 0.06507397069667138, 0.06507397069667123, 0.48971269294890085],
    linf = [2.383463161765847, 4.0791883314039605, 4.07918833140396, 16.246070713311475],
    tspan = (0.0, 0.05))
  end
end


@testset "Displaying components 2D" begin
  @test_nowarn include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"))

  # test both short and long printing formats
  @test_nowarn show(mesh); println()
  @test_nowarn println(mesh)
  @test_nowarn display(mesh)

  @test_nowarn show(equations); println()
  @test_nowarn println(equations)
  @test_nowarn display(equations)

  @test_nowarn show(solver); println()
  @test_nowarn println(solver)
  @test_nowarn display(solver)

  @test_nowarn show(solver.basis); println()
  @test_nowarn println(solver.basis)
  @test_nowarn display(solver.basis)

  @test_nowarn show(solver.mortar); println()
  @test_nowarn println(solver.mortar)
  @test_nowarn display(solver.mortar)

  @test_nowarn show(semi); println()
  @test_nowarn println(semi)
  @test_nowarn display(semi)

  @test_nowarn show(summary_callback); println()
  @test_nowarn println(summary_callback)
  @test_nowarn display(summary_callback)

  @test_nowarn show(amr_controller); println()
  @test_nowarn println(amr_controller)
  @test_nowarn display(amr_controller)

  @test_nowarn show(amr_callback); println()
  @test_nowarn println(amr_callback)
  @test_nowarn display(amr_callback)

  @test_nowarn show(stepsize_callback); println()
  @test_nowarn println(stepsize_callback)
  @test_nowarn display(stepsize_callback)

  @test_nowarn show(save_solution); println()
  @test_nowarn println(save_solution)
  @test_nowarn display(save_solution)

  @test_nowarn show(analysis_callback); println()
  @test_nowarn println(analysis_callback)
  @test_nowarn display(analysis_callback)

  @test_nowarn show(alive_callback); println()
  @test_nowarn println(alive_callback)
  @test_nowarn display(alive_callback)

  @test_nowarn println(callbacks)
end

# Only run extended tests if environment variable is set
if haskey(ENV, "TRIXI_TEST_EXTENDED") && lowercase(ENV["TRIXI_TEST_EXTENDED"]) in ("1", "on", "yes")
  @testset "Examples (long execution time)" begin
    @test_nowarn test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_shockcapturing_mortar.jl"))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D

end #module
