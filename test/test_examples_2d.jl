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
  @testset "elixir_advection_basic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.144681765639205e-6],
      linf = [6.437440532547356e-5])
  end

  @testset "elixir_advection_mortar.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
    l2   = [0.022356422238096973],
    linf = [0.5043638249003257])
  end

  @testset "elixir_advection_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
    l2   = [0.12533080510721473],
    linf = [0.9999802982947753])
  end


  @testset "elixir_hyp_diff_llf.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_llf.jl"),
      l2   = [0.0001568775181745306, 0.001025986772217103, 0.0010259867722170538],
      linf = [0.0011986956378152724, 0.006423873516111733, 0.006423873516110845])
  end

  @testset "elixir_hyp_diff_harmonic_nonperiodic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_harmonic_nonperiodic.jl"),
      l2   = [8.61813235543625e-8, 5.619399844542781e-7, 5.6193998447443e-7],
      linf = [1.124861862180196e-6, 8.622436471039663e-6, 8.622436470151484e-6])
  end

  @testset "elixir_hyp_diff_nonperiodic.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_nonperiodic.jl"),
      l2   = [8.523077653955306e-6, 2.8779323653065056e-5, 5.4549427691297846e-5],
      linf = [5.5227409524905013e-5, 0.0001454489597927185, 0.00032396328684569653])
  end

  @testset "elixir_hyp_diff_upwind.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hyp_diff_upwind.jl"),
      l2   = [5.868147556385677e-6, 3.805179273239753e-5, 3.805179273248075e-5],
      linf = [3.7019654930525725e-5, 0.00021224229433514097, 0.00021224229433514097])
  end


  @testset "elixir_euler_source_terms.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
    l2   = [8.517783186497567e-7, 1.2350199409361865e-6, 1.2350199409828616e-6, 4.277884398786315e-6],
    linf = [8.357934254688004e-6, 1.0326389653148027e-5, 1.0326389654924384e-5, 4.4961900057316484e-5])
  end

  @testset "elixir_euler_density_wave.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
    l2   = [0.0010600778457965205, 0.00010600778457646603, 0.0002120155691588112, 2.6501946142012653e-5],
    linf = [0.006614198043407127, 0.0006614198043931596, 0.001322839608845383, 0.00016535495117153687],
    tspan = (0.0, 0.5))
  end

  @testset "elixir_euler_nonperiodic.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic.jl"),
    l2   = [2.3652137675654753e-6, 2.1386731303685556e-6, 2.138673130413185e-6, 6.009920290578574e-6],
    linf = [1.4080448659026246e-5, 1.7581818010814487e-5, 1.758181801525538e-5, 5.9568540361709665e-5])
  end

  @testset "elixir_euler_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
    l2   = [0.061733846713578594, 0.05020086119442834, 0.05020836856347214, 0.2259064869636338],
    linf = [0.29894122391731826, 0.30853631977725215, 0.3084722538869674, 1.0652455597305965])
  end

  @testset "elixir_euler_blast_wave_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_shockcapturing.jl"),
    l2   = [0.13575932799459445, 0.11346025131402862, 0.11346028941202581, 0.33371846538168354],
    linf = [1.4662633480487193, 1.3203905049492335, 1.320390504949303, 1.8131376065886553],
    tspan = (0.0, 0.13))
  end

  @testset "elixir_euler_weak_blast_wave_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weak_blast_wave_shockcapturing.jl"),
    l2   = [0.053797693352771236, 0.0469609422046655, 0.04696357535470453, 0.19685219525959569],
    linf = [0.18540098690235163, 0.2402949901937739, 0.23266805976720523, 0.6874635927547934])
  end

  @testset "elixir_euler_blast_wave_shockcapturing_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_shockcapturing_amr.jl"),
    l2   = [0.6778339184192986, 0.28136085729167076, 0.2813607687129121, 0.7202946425475186],
    linf = [2.8891939545999277, 1.8038083274644838, 1.8036523839220984, 3.0363712085327177],
    tspan = (0.0, 1.0))
  end

  @testset "elixir_euler_sedov_blast_wave_shockcapturing_amr.jl with tend = 1.0" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_shockcapturing_amr.jl"),
    l2   = [0.48179128651635356, 0.16552908046011455, 0.16553045844776362, 0.6182628255460497],
    linf = [2.4847876521233907, 1.2814307117459813, 1.2814769220593392, 6.474196250771773],
    tspan = (0.0, 1.0))
  end

  @testset "elixir_euler_blob_split_shockcapturing_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_split_shockcapturing_amr.jl"),
    l2   = [0.2079529146644449, 1.2165976525172113, 0.10497525531751525, 5.343396906455776],
    linf = [14.746412579562035, 73.35401826630807, 7.945659812348401, 299.28120847051116],
    tspan = (0.0, 0.12))
  end

  @testset "elixir_euler_khi_amr.jl with tend = 0.2" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi_amr.jl"),
    l2   = [0.0019809356303815313, 0.006538462481807526, 0.004737804472678921, 0.0050181776990539505],
    linf = [0.016342197215556853, 0.03993613023503173, 0.015293069044755532, 0.024177402362647094],
    tspan = (0.0, 0.2))
  end

  @testset "elixir_euler_vortex.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
    l2   = [3.6342636871275523e-6, 0.0032111366825032443, 0.0032111479254594345, 0.004545714785045611],
    linf = [7.903587114788113e-5, 0.030561314311228993, 0.030502600162385596, 0.042876297246817074])
  end

  @testset "elixir_euler_vortex_mortar.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
    l2   = [2.120307461394424e-6, 2.7929229084570266e-5, 3.759342242369596e-5, 8.813646673773311e-5],
    linf = [5.9320459189771135e-5, 0.0007491265403041236, 0.0008165690047976515, 0.0022122638048145404])
  end

  @testset "elixir_euler_vortex_mortar_split.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.1203693476896995e-6, 2.8053512416422296e-5, 3.76179445622429e-5, 8.840787521479401e-5],
    linf = [5.9005667252809424e-5, 0.0007554116730550398, 0.00081660478740464, 0.002209016304192346])
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_central" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.120307461409829e-6, 2.7929229084583212e-5, 3.759342242369501e-5, 8.813646673812448e-5],
    linf = [5.932045918888296e-5, 0.0007491265403021252, 0.0008165690047987617, 0.002212263804818093],
    volume_flux = flux_central)
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_shima_etal" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.120103291509122e-6, 2.805652562691104e-5, 3.759500428816484e-5, 8.841374592860891e-5],
    linf = [5.934103184424e-5, 0.0007552316820342853, 0.0008152449048961508, 0.002206987374638203],
    volume_flux = flux_shima_etal)
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_ranocha" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.1201032806889955e-6, 2.8056528074361895e-5, 3.759500957406334e-5, 8.841379428954133e-5],
    linf = [5.934027760512439e-5, 0.0007552314317718078, 0.0008152450117491217, 0.0022069976113101575],
    volume_flux = flux_ranocha)
  end

  @testset "elixir_euler_vortex_split_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_split_shockcapturing.jl"),
    l2   = [3.80342739421474e-6, 5.561118953968859e-5, 5.564042529709319e-5, 0.0001570628548096201],
    linf = [8.491382365727329e-5, 0.0009602965158113097, 0.0009669978616948516, 0.0030750353269972663])
  end

  @testset "elixir_euler_vortex_mortar_split_shockcapturing.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split_shockcapturing.jl"),
    l2   = [2.1203693476896995e-6, 2.8053512416422296e-5, 3.76179445622429e-5, 8.840787521479401e-5],
    linf = [5.9005667252809424e-5, 0.0007554116730550398, 0.00081660478740464, 0.002209016304192346])
  end

  @testset "elixir_mhd_alfven_wave.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
    l2   = [0.00011134312019581907, 5.880417517656501e-6, 5.880417517683334e-6, 8.433533326833135e-6, 1.2944026635567339e-6, 1.2259080543012733e-6, 1.2259080543038862e-6, 1.8334999489680995e-6, 8.098795948637635e-7],
    linf = [0.0002678907090871707, 1.6257637592484442e-5, 1.6257637592095864e-5, 2.7343412701746894e-5, 5.327954748168828e-6, 8.10079419122367e-6, 8.100794191445715e-6, 1.2083599637696674e-5, 4.179907421413125e-6])
  end

  @testset "elixir_mhd_alfven_wave_mortar.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
    l2   = [4.6108151315202035e-6, 1.6897860606321754e-6, 1.6208236429504275e-6, 1.6994662614575904e-6, 1.486435064660995e-6, 1.3875465211720615e-6, 1.3411325436690753e-6, 1.7155153011375413e-6, 9.813872476368202e-7],
    linf = [3.5225667207927636e-5, 1.5349379665866025e-5, 1.4264328575347429e-5, 1.4421439547898651e-5, 7.744170905765735e-6, 1.0187833250130396e-5, 9.861911995590056e-6, 1.6018139446766222e-5, 5.563892853177171e-6],
    tspan = (0.0, 1.0))
  end

  @testset "elixir_mhd_ec.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
    l2   = [0.03628315925311581, 0.04301306535453907, 0.042998910996002976, 0.025746791646914315, 0.1620587870592711, 0.01745580631201365, 0.01745656644392971, 0.02688212902288343, 0.00014263322984147517],
    linf = [0.23504901239438747, 0.31563591777956146, 0.3094412744514615, 0.21177505529310434, 0.9738775041875032, 0.09120517132559702, 0.0919645047337756, 0.15691668358334432, 0.0035581030835232378])
  end

  @testset "elixir_mhd_orszag_tang_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang_amr.jl"),
    l2   = [0.10792892784381206, 0.20179133761722504, 0.22962184589189918, 0.0, 0.2993794211703735, 0.1570642713959375, 0.24295137209226844, 0.0, 0.012333314953338195],
    linf = [0.5613440997713406, 0.5101525577854369, 0.6592905328573747, 0.0, 0.9883261765401286, 0.401691133525005, 0.6750421877831418, 0.0, 0.20250593792711205],
    tspan = (0.0, 0.06))
  end

  @testset "elixir_mhd_rotor_amr.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor_amr.jl"),
    l2   = [0.3706256586052198, 0.8397913575257775, 0.8061150883165829, 0.0, 0.9158524987760742, 0.11469774220979392, 0.14167686867748344, 0.0, 0.01843234514008023],
    linf = [4.909133664840393, 9.429512581671567, 7.668678084481505, 0.0, 10.777238495921754, 1.294018631465027, 1.4569385394302714, 0.0, 0.2734224559555755],
    tspan = (0.0, 0.02))
  end

  @test_skip test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave_amr.jl"), tspan=(0.0, 1.0e-4))

  @testset "elixir_euler_gravity_jeans_instability.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_gravity_jeans_instability.jl"),
    l2   = [21174.129216837846, 978.8980277332366, 4.723889542064964e-6, 52935.30182880739],
    linf = [29951.765533944592, 1388.65770524552, 2.0555889477917298e-5, 74879.49312048778],
    tspan = (0.0, 0.6))
  end

  @testset "elixir_euler_gravity_sedov_blast_wave.jl" begin
  test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_gravity_sedov_blast_wave.jl"),
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
    @test_nowarn test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_mortar_split_shockcapturing.jl"))
    @test_nowarn test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_split_shockcapturing_amr.jl"))
    @test_nowarn test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi.jl"))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D

end #module
