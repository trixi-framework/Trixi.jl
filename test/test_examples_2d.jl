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
      l2   = [0.06172864640680411, 0.050194807377561185, 0.050202324800403486, 0.22588683333743503],
      linf = [0.29813572480585526, 0.3069377110825767, 0.306807092333435, 1.062952871675828])
  end

  @testset "elixir_euler_weak_blast_wave_shockcapturing.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weak_blast_wave_shockcapturing.jl"),
      l2   = [0.053797946432602085, 0.04696120828935379, 0.04696384063506395, 0.19685320969570913],
      linf = [0.18540158860112732, 0.24029373364236004, 0.23267525584314722, 0.6874555954921888])
  end

  @testset "elixir_euler_blast_wave_shockcapturing.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_shockcapturing.jl"),
      l2   = [0.13572637700619994, 0.11345063094668695, 0.1134506678304754, 0.333697254342641],
      linf = [1.4672307469648924, 1.3207761777665028, 1.3207761777665479, 1.8104501622316787],
      tspan = (0.0, 0.13))
  end

  @testset "elixir_euler_blast_wave_shockcapturing_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_shockcapturing_amr.jl"),
      l2   = [0.6779856619001925, 0.2814963219016726, 0.2814961545188141, 0.7227078877591626],
      linf = [2.8903767693905342, 1.8018637904659396, 1.801813163681165, 3.0522925471933595],
      tspan = (0.0, 1.0))
  end

  @testset "elixir_euler_sedov_blast_wave_shockcapturing_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_shockcapturing_amr.jl"),
      l2   = [0.48015752260582717, 0.16490465208329974, 0.16490470637844776, 0.6182032928050549],
      linf = [2.4758679663634737, 1.2774510137145505, 1.27745426873474, 6.474450939003187],
      tspan = (0.0, 1.0))
  end

  @testset "elixir_euler_blob_shockcapturing_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_shockcapturing_amr.jl"),
      l2   = [0.2012143467980036, 1.1813241716700988, 0.10144725208346557, 5.230607564921326],
      linf = [14.111578610092542, 71.21944410118338, 7.304666476530256, 291.9385076318331],
      tspan = (0.0, 0.12))
  end

  @testset "elixir_euler_khi_shockcapturing_amr.jl with tend = 0.2" begin
    if Threads.nthreads() == 1
      # This example uses random numbers to generate the initial condition.
      # Hence, we can only check "errors" if everything is made reproducible.
      # However, that's not enough to ensure reproducibility since the stream
      # of random numbers is not guaranteed to be the same across different
      # minor versions of Julia.
      # See https://github.com/trixi-framework/Trixi.jl/issues/232#issuecomment-709738400
      test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi_shockcapturing_amr.jl"),
        l2   = [0.0019809356303815313, 0.006538462481807526, 0.004737804472678921, 0.0050181776990539505],
        linf = [0.016342197215556853, 0.03993613023503173, 0.015293069044755532, 0.024177402362647094],
        tspan = (0.0, 0.2))
    else
      test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi_shockcapturing_amr.jl"),
        tspan = (0.0, 0.2))
    end
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

  @testset "elixir_euler_vortex_shockcapturing.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
      l2   = [3.80342739421474e-6, 5.561118953968859e-5, 5.564042529709319e-5, 0.0001570628548096201],
      linf = [8.491382365727329e-5, 0.0009602965158113097, 0.0009669978616948516, 0.0030750353269972663])
  end

  @testset "elixir_euler_vortex_mortar_shockcapturing.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_shockcapturing.jl"),
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
      l2   = [0.036282196554965125, 0.04301109667609445, 0.04299694569681459, 0.025746537317269044, 0.16205093281676458, 0.01745444180120457, 0.01745520557754369, 0.026880604347203515, 0.00014126194508100613],
      linf = [0.2350953403606807, 0.31558387317731673, 0.3093303919451385, 0.21173804432368912, 0.9727624987129335, 0.09099668342879141, 0.09175183163544531, 0.15718993346004728, 0.0034956827655864974])
  end

  @testset "elixir_mhd_orszag_tang_shockcapturing_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang_shockcapturing_amr.jl"),
      l2   = [0.1078977432446604, 0.20175224352695822, 0.22961400192211595, 0.0, 0.29936249920620345, 0.15703050049823944, 0.24293670982687385, 0.0, 0.011948713883663134],
      linf = [0.5607164640963931, 0.510068211143, 0.6637697474103158, 0.0, 0.989841573616437, 0.404125796210551, 0.675821346369299, 0.0, 0.18563744518118758],
      tspan = (0.0, 0.06))
  end

  @testset "elixir_mhd_rotor_shockcapturing_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor_shockcapturing_amr.jl"),
      l2   = [0.3705710836691761, 0.8391184023362467, 0.8065525179081886, 0.0, 0.915470179159569, 0.11451033549419987, 0.1417509058916749, 0.0, 0.01840151588323637],
      linf = [4.7393103018367, 9.426966061390797, 7.618589342811914, 0.0, 10.726660868768525, 1.1883141549085037, 1.4532105708358205, 0.0, 0.2631428555105575],
      tspan = (0.0, 0.02))
  end

  @test_skip test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave_shockcapturing_amr.jl"), tspan=(0.0, 1.0e-4))


  @testset "elixir_eulergravity_jeans_instability.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_jeans_instability.jl"),
      l2   = [21174.12949565436, 978.8960762624803, 3.89301153085966e-6, 52935.302525851395],
      linf = [29951.76592380926, 1388.654939246432, 1.4848956710887516e-5, 74879.49409445375],
      tspan = (0.0, 0.6))
  end

  @testset "elixir_eulergravity_sedov_blast_wave_shockcapturing_amr.jl" begin
    test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_sedov_blast_wave_shockcapturing_amr.jl"),
      l2   = [0.04631599064839133, 0.06508178881706861, 0.06508178881706848, 0.48967401029970314],
      linf = [2.3874843337593776, 4.07876384374792, 4.07876384374792, 16.23914384809855],
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
    @test_nowarn test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_mortar_shockcapturing.jl"))
    @test_nowarn test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_shockcapturing_amr.jl"))
    @test_nowarn test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_khi_shockcapturing.jl"))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D

end #module
