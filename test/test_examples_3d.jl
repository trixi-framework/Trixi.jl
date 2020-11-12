module TestExamples3D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

@testset "3D" begin

# Run basic tests
@testset "Examples 3D" begin
  @testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.00015975755208652597],
      linf = [0.0015038732976652147])
  end

  @testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [0.0001780001287314664],
      linf = [0.0014520752637396939])
  end

  # TODO Taal: create separate elixirs for ICs/BCs etc. to keep `basic` simple
  @testset "elixir_advection_basic.jl with initial_condition_sin" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.002727293067654415],
      linf = [0.024833049753677727],
      initial_condition=Trixi.initial_condition_sin)
  end

  @testset "elixir_advection_basic.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [7.033186506921888e-16],
      linf = [2.6645352591003757e-15],
      initial_condition=initial_condition_constant)
  end

  @testset "elixir_advection_basic.jl with initial_condition_linear_z and periodicity=false" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [5.887699520794518e-16],
      linf = [6.217248937900877e-15],
      initial_condition=Trixi.initial_condition_linear_z,
      boundary_conditions=Trixi.boundary_condition_linear_z, periodicity=false)
  end

  @testset "elixir_advection_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
      l2   = [0.0018461529502663268],
      linf = [0.01785420966285467])
  end

  @testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [9.773852895157622e-6],
      linf = [0.0005853874124926162])
  end


  # Hyperbolic diffusion
  @testset "elixir_hypdiff_lax_friedrichs.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
      l2   = [0.001530331609036682, 0.011314177033289238, 0.011314177033289402, 0.011314177033289631],
      linf = [0.02263459033909354, 0.10139777904683545, 0.10139777904683545, 0.10139777904683545],
      initial_refinement_level=2)
  end

  @testset "elixir_hypdiff_lax_friedrichs.jl with surface_flux=flux_upwind)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
      l2   = [0.0015377731806850128, 0.01137685274151801, 0.011376852741518175, 0.011376852741518494],
      linf = [0.022715420630041172, 0.10183745338964201, 0.10183745338964201, 0.1018374533896429],
      initial_refinement_level=2, surface_flux=flux_upwind)
  end

  @testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
      l2   = [0.00022868320512754316, 0.0007974309948540525, 0.0015035143230654987, 0.0015035143230655293],
      linf = [0.0016405001653623241, 0.0029870057159104594, 0.009410031618285686, 0.009410031618287462])
  end


  # Compressible Euler
  @testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.01032310150257373, 0.009728768969448439, 0.009728768969448494, 0.009728768969448388, 0.015080412597559597],
      linf = [0.034894790428615874, 0.033835365548322116, 0.033835365548322116, 0.03383536554832034, 0.06785765131417065])
  end

  @testset "elixir_euler_source_terms.jl with split_form" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.010323101502574773, 0.009728768969449282, 0.009728768969449344, 0.009728768969449401, 0.015080412597560888],
      linf = [0.0348947904286212, 0.03383536554832034, 0.03383536554831723, 0.03383536554831679, 0.0678576513141671],
      volume_integral=VolumeIntegralFluxDifferencing(flux_central))
  end

  @testset "elixir_euler_eoc_test.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_eoc_test.jl"),
      l2   = [0.0003637241020254405, 0.0003955570866382718, 0.0003955570866383613, 0.00039555708663834417, 0.0007811613481640202],
      linf = [0.0024000660244674066, 0.0029635410025339315, 0.0029635410025292686, 0.002963541002525938, 0.007191437359396424])
  end

  @testset "elixir_euler_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_mortar.jl"),
      l2   = [0.0019011097544691046, 0.0018289464161846331, 0.0018289464161847266, 0.0018289464161847851, 0.0033547668596639966],
      linf = [0.011918626829790169, 0.011808582902362641, 0.01180858290237552, 0.011808582902357312, 0.024648094686513744])
  end

  @testset "elixir_euler_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_amr.jl"),
      l2   = [0.0038281919675174224, 0.003828191967517416, 0.003828191967517415, 0.003828191967517416, 0.0057422879512759525],
      linf = [0.07390148817126874, 0.07390148817126896, 0.07390148817126896, 0.07390148817126874, 0.110852232256903],
      tspan=(0.0, 0.1))
  end

  @testset "elixir_euler_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
      l2   = [0.000349497725802709, 0.031333864159293845, 0.03133386415929371, 0.04378595044851934, 0.015796652303527357],
      linf = [0.0013934750216293423, 0.07242017454880123, 0.07242017454880156, 0.12796560115483002, 0.07680757651078807],
      tspan = (0.0, 0.5))
  end

  @testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.02570137197844877, 0.016179934130642552, 0.01617993413064253, 0.016172648598753545, 0.09261669328795467],
      linf = [0.3954458125573179, 0.26876916180359345, 0.26876916180359345, 0.26933123042178553, 1.3724137121660251])
  end

  @testset "elixir_euler_shockcapturing.jl with initial_condition_sedov_blast_wave" begin
    # OBS! This setup does not run longer but crashes (also the parameters do not make sense) -> only for testing the IC!
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.03627061709701234, 0.05178777768691398, 0.05178777768691404, 0.051787777686913984, 0.2308467334945545],
      linf = [0.9542995612960261, 1.2654683412445416, 1.2654683412445418, 1.2654683412445418, 12.805752164787227],
      initial_condition=initial_condition_sedov_blast_wave, cfl=0.7, alpha_max=1.0, tspan=(0.0, 0.1))
  end

  @testset "elixir_euler_shockcapturing.jl with initial_condition_sedov_self_gravity" begin
    # OBS! This setup does not run longer but crashes (also the parameters do not make sense) -> only for testing the IC!
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.048222481449477446, 0.0517877776783649, 0.051787777678364795, 0.05178777767836493, 0.230846733492145],
      linf = [0.954296651568088, 1.2654683413909855, 1.2654683413909855, 1.2654683413909857, 12.805752164822744],
      initial_condition=initial_condition_sedov_self_gravity, cfl=0.7, alpha_max=1.0, tspan=(0.0, 0.1))
  end

  @testset "elixir_euler_shockcapturing_amr.jl" begin
    # OBS! This setup does not make much practical sense. It is only added to exercise the
    # `sedov_self_gravity` AMR indicator, which in its original configuration is too expensive for
    # CI testing
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_amr.jl"),
      l2   = [0.02217299067704248, 0.012771561294571411, 0.01277156129457143, 0.012770635779336643, 0.08091898488262424],
      linf = [0.4047819603427084, 0.27493532130155474, 0.2749353213015551, 0.2749304638368023, 1.4053942765487641],
      maxiters=10)
  end

  @testset "elixir_euler_density_pulse.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_pulse.jl"),
      l2   = [0.057196526814004715, 0.057196526814004715, 0.05719652681400473, 0.057196526814004736, 0.08579479022100575],
      linf = [0.27415246703018203, 0.2741524670301829, 0.2741524670301827, 0.27415246703018226, 0.41122870054527816])
  end

  @testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.02526341317987378, 0.016632068583699623, 0.016632068583699623, 0.01662548715216875, 0.0913477018048886],
      linf = [0.4372549540810414, 0.28613118232798984, 0.28613118232799006, 0.28796686065271876, 1.5072828647309124])
  end

  @testset "elixir_euler_ec.jl with initial_condition=initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [4.183721551616214e-16, 6.059779958716338e-16, 4.916596221090319e-16, 9.739943366304456e-16, 3.7485908743251566e-15],
      linf = [2.4424906541753444e-15, 3.733124920302089e-15, 4.440892098500626e-15, 5.329070518200751e-15, 2.4868995751603507e-14],
      initial_condition=initial_condition_constant)
  end

  @testset "elixir_euler_ec.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.025265721172813106, 0.016649800693500427, 0.01664980069350042, 0.01664379306708522, 0.09137248646784184],
      linf = [0.4373399329742198, 0.28434487167605427, 0.28434487167605427, 0.28522678968890774, 1.532471676033761],
      surface_flux=flux_chandrashekar, volume_flux=flux_chandrashekar)
  end

  @testset "elixir_euler_ec.jl with flux_kennedy_gruber" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.025280033869871984, 0.016675487948639846, 0.016675487948639853, 0.016668992714991282, 0.091455613470441],
      linf = [0.43348628145015766, 0.28853549062014217, 0.28853549062014217, 0.2903943042772536, 1.5236557526482426],
      surface_flux=flux_kennedy_gruber, volume_flux=flux_kennedy_gruber)
  end

  @testset "elixir_euler_ec.jl with flux_shima_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.025261716925811403, 0.016637655557848952, 0.01663765555784895, 0.01663105921013437, 0.09136239054024566],
      linf = [0.43692416928732536, 0.28622033209064734, 0.28622033209064746, 0.2881197143457632, 1.506534270303663],
      surface_flux=flux_shima_etal, volume_flux=flux_shima_etal)
  end

  @testset "elixir_euler_blob_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_amr.jl"),
      l2   = [0.04867856452253151, 0.2640486962336911, 0.0354927658652858, 0.03549276586528571, 1.0777274757408568],
      linf = [9.558543313792217, 49.4518309553356, 10.319859082570309, 10.319859082570487, 195.1066220797401],
      tspan = (0.0, 0.2))
  end


  # MHD
  @testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.01728564856700721, 0.017770558428794502, 0.01777055842879448, 0.017772303855024686, 0.07402251095395435, 0.010363317528939847, 0.010363317528939873, 0.010365251266968387, 0.00020781240461593321],
      linf = [0.2648387662456203, 0.33478411844879813, 0.3347841184487984, 0.3698107321074581, 1.2338949711031062, 0.09857295382870013, 0.09857295382870068, 0.10426497383213318, 0.008020325762909617])
  end

  @testset "elixir_mhd_ec.jl with initial_condition=initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [5.447723083700851e-16, 2.617532329387129e-15, 3.879181146896324e-15, 2.9617662710806352e-15, 2.684011421945918e-14, 1.371966389577962e-15, 1.3058638553193934e-15, 1.2475529686776005e-15, 1.4573570237027504e-15],
      linf = [3.9968028886505635e-15, 1.4155343563970746e-14, 2.2898349882893854e-14, 1.8707257964933888e-14, 2.2737367544323206e-13, 1.2878587085651816e-14, 7.327471962526033e-15, 9.325873406851315e-15, 8.568189425129782e-15],
      atol = 1000*eps(),
      initial_condition=initial_condition_constant)
  end

  @testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.0038734936431132137, 0.0090374494786511, 0.004173235737356234, 0.011605032955462995, 0.006247939160992442, 0.009226153647367723, 0.003460561919838917, 0.011684984122517875, 0.002201096128779557],
      linf = [0.012630239144833078, 0.03265663914470475, 0.01291368559476544, 0.04444329719972474, 0.027974787995271644, 0.03453507441133391, 0.01022512252706076, 0.04498328542685666, 0.009861640501698054])
  end

  @testset "elixir_mhd_alfven_wave_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
      l2   = [0.0021484230386442484, 0.006826974752320685, 0.0030655172071723945, 0.008736026827927016, 0.005159744590177398, 0.007158578094946651, 0.0028289753083129443, 0.008815093448251163, 0.0022268140033108634],
      linf = [0.013181862339541217, 0.05433825341881836, 0.02085489712458487, 0.05947341475014939, 0.03171319598340849, 0.05439174028778537, 0.017933784069765202, 0.06036923226300564, 0.012816582427655984],
      tspan = (0.0, 0.25))
  end

  @testset "elixir_mhd_orszag_tang.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
      l2   = [0.0043911605686459704, 0.041447356655813394, 0.04150129977673066, 0.041503536105048366, 0.03693119824334232, 0.02112559892198415, 0.03295606821170978, 0.03296235617354949, 6.360380099124839e-6],
      linf = [0.01789383890583951, 0.0848675349327856, 0.08910602912506882, 0.08491879187575965, 0.10444596146695251, 0.05381953967385888, 0.08847783436169666, 0.07784630781912419, 8.236241065117021e-5],
      tspan = (0.0, 0.06))
  end


  # Compressible Euler with self-gravity
  @testset "elixir_eulergravity_eoc_test.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_eoc_test.jl"),
      l2   = [0.0004276779201667428, 0.00047204222332596204, 0.00047204222332608705, 0.0004720422233259819, 0.0010987026250960728],
      linf = [0.003496616916238704, 0.003764418290373106, 0.003764418290377103, 0.0037644182903766588, 0.008370424899251105],
      resid_tol = 1.0e-4, tspan = (0.0, 0.2))
  end
end


@testset "Displaying components 3D" begin
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

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 3D

end #module
