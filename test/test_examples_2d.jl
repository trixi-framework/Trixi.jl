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
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.144681765639205e-6],
      linf = [6.437440532547356e-5])
  end

  @testset "elixir_advection_basic.jl with polydeg=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.05264106093598111],
      linf = [0.08754218386076518],
      polydeg=1)
  end

  @testset "elixir_advection_timeintegration.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [9.144681765639205e-6],
      linf = [6.437440532547356e-5])
  end

  @testset "elixir_advection_timeintegration.jl with carpenter_kennedy_erk43" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [8.908962577028364e-6],
      linf = [6.969419032576418e-5],
      ode_algorithm=Trixi.CarpenterKennedy2N43(),
      cfl = 1.0)
  end

  @testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk94" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [7.932405161658336e-6],
      linf = [6.509399993848142e-5],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar94())
  end

  @testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk32" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [0.00440542760645958],
      linf = [0.012549162970726613],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar32())
  end

  @testset "elixir_advection_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
      l2   = [0.001613367076589483],
      linf = [0.013960195840607481])
  end

  @testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [0.010844189678803203],
      linf = [0.0491178481591637])
  end

  @testset "elixir_advection_amr_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
      l2   = [0.008016815805080098],
      linf = [0.04229543866599861])
  end

  @testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [1.2148032444677485e-5],
      linf = [6.495644794757283e-5])
  end

  @testset "elixir_advection_callbacks.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_callbacks.jl"))
  end


  @testset "elixir_hypdiff_lax_friedrichs.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
      l2   = [0.0001568775108748819, 0.0010259867353406083, 0.0010259867353406382],
      linf = [0.0011986956416590866, 0.006423873516411938, 0.006423873516411938])
  end

  @testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_harmonic_nonperiodic.jl"),
      l2   = [8.618132353932638e-8, 5.619399844708813e-7, 5.619399845476024e-7],
      linf = [1.124861862326869e-6, 8.622436471483752e-6, 8.622436469707395e-6])
  end

  @testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
      l2   = [8.523077654037775e-6, 2.877932365308637e-5, 5.454942769137812e-5],
      linf = [5.484978959957587e-5, 0.00014544895979200218, 0.000324491268921534])
  end

  @testset "elixir_hypdiff_upwind.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_upwind.jl"),
      l2   = [5.868147556488962e-6, 3.8051792732628014e-5, 3.8051792732620214e-5],
      linf = [3.70196549871471e-5, 0.0002072058411455302, 0.00020720584114464202])
  end


  @testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [8.517783186497567e-7, 1.2350199409361865e-6, 1.2350199409828616e-6, 4.277884398786315e-6],
      linf = [8.357934254688004e-6, 1.0326389653148027e-5, 1.0326389654924384e-5, 4.4961900057316484e-5])
  end

  @testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.001060077845747576, 0.00010600778457107525, 0.00021201556914875742, 2.6501946139091318e-5],
      linf = [0.0065356386867677085, 0.0006535638688170142, 0.0013071277374487877, 0.0001633909674296774],
      tspan = (0.0, 0.5))
  end

  @testset "elixir_euler_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic.jl"),
      l2   = [2.3652137675654753e-6, 2.1386731303685556e-6, 2.138673130413185e-6, 6.009920290578574e-6],
      linf = [1.4080448659026246e-5, 1.7581818010814487e-5, 1.758181801525538e-5, 5.9568540361709665e-5])
  end

  @testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.06159341742582756, 0.05012484425381723, 0.05013298724507752, 0.22537740506116724],
      linf = [0.29912627861573327, 0.30886767304359375, 0.3088108573487326, 1.0657556075017878])
  end

  @testset "elixir_euler_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [0.13910202327088322, 0.11538722576277083, 0.1153873048510009, 0.3387876385945495],
      linf = [1.454418325889352, 1.3236875559310013, 1.323687555933169, 1.8225476335086368],
      maxiters=30)
  end

  @testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.05365734539276933, 0.04683903386565478, 0.04684207891980008, 0.19632055541821553],
      linf = [0.18542234326379825, 0.24074440953554058, 0.23261143887822433, 0.687464986948263])
  end

  @testset "elixir_euler_blast_wave_amr.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
    l2   = [0.6776486969229696, 0.2813026529898539, 0.2813025645101231, 0.7174702524881597],
    linf = [2.8939055423031546, 1.7997630098946877, 1.7997118659969253, 3.0341223482585686],
    tspan = (0.0, 1.0))
  end

  @testset "elixir_euler_sedov_blast_wave.jl with tend = 1.0" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
    l2   = [0.4820048896322639, 0.16556563003698888, 0.16556563003698901, 0.643610807739157],
    linf = [2.485752556439829, 1.2870638985941658, 1.2870638985941667, 6.474544663221404],
    tspan = (0.0, 1.0))
  end

  @testset "elixir_euler_blob_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_mortar.jl"),
      l2   = [0.22114610074017435, 0.6275613030540599, 0.24325218693791564, 2.925865235621878],
      linf = [10.524011747446043, 27.512527136693347, 9.454054943042742, 97.53367336970214],
      tspan = (0.0, 0.5))
  end

  @testset "elixir_euler_blob_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_amr.jl"),
      l2   = [0.2016728420174888, 1.1836138789789359, 0.10165086496270354, 5.237367755805095],
      linf = [14.085819993255987, 71.07473800830421, 7.366144023918916, 297.24197965204814],
      tspan = (0.0, 0.12))
  end

  @testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    if Threads.nthreads() == 1
      # This example uses random numbers to generate the initial condition.
      # Hence, we can only check "errors" if everything is made reproducible.
      # However, that's not enough to ensure reproducibility since the stream
      # of random numbers is not guaranteed to be the same across different
      # minor versions of Julia.
      # See https://github.com/trixi-framework/Trixi.jl/issues/232#issuecomment-709738400
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability.jl"),
        l2   = [0.002046615463716511, 0.002862576343897973, 0.001971146183422579, 0.004817029337018751],
        linf = [0.024299256322982465, 0.01620011715132652, 0.009869197749689947, 0.02060000394920891],
        tspan = (0.0, 0.2))
    else
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability.jl"),
        tspan = (0.0, 0.2))
    end
  end

  @testset "elixir_euler_kelvin_helmholtz_instability_amr.jl" begin
    if Threads.nthreads() == 1
      # This example uses random numbers to generate the initial condition.
      # Hence, we can only check "errors" if everything is made reproducible.
      # However, that's not enough to ensure reproducibility since the stream
      # of random numbers is not guaranteed to be the same across different
      # minor versions of Julia.
      # See https://github.com/trixi-framework/Trixi.jl/issues/232#issuecomment-709738400
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability_amr.jl"),
        l2   = [0.001653490458693617, 0.0023814551690212226, 0.0013742646130843919, 0.0031589243386909585],
        linf = [0.022479473484114054, 0.015056172762090259, 0.0070761455651367836, 0.01461791479513419],
        tspan = (0.0, 0.2))
    else
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability_amr.jl"),
        tspan = (0.0, 0.2))
    end
  end

  @testset "elixir_euler_vortex.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
    l2   = [3.6343138447409784e-6, 0.0032111379843728876, 0.0032111482778261658, 0.004545715889714643],
    linf = [7.901869034399045e-5, 0.030511158864742205, 0.030451936462313256, 0.04361908901631395])
  end

  @testset "elixir_euler_vortex_mortar.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
    l2   = [2.1202421511973067e-6, 2.7929028341308907e-5, 3.7593065314592924e-5, 8.813423453465327e-5],
    linf = [5.93205509794581e-5, 0.0007486675478352023, 0.0008175405566226424, 0.002212267888996422])
  end

  @testset "elixir_euler_vortex_mortar_split.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.1203040671963692e-6, 2.8053312800289536e-5, 3.761758762899687e-5, 8.840565162128428e-5],
    linf = [5.900575985384737e-5, 0.0007547236106317801, 0.000817616344069072, 0.0022090204216524967])
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_central" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.1202421512026147e-6, 2.7929028341288412e-5, 3.759306531457842e-5, 8.813423453452753e-5],
    linf = [5.932055097812583e-5, 0.0007486675478027838, 0.0008175405566221983, 0.0022122678889928693],
    volume_flux = flux_central)
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_shima_etal" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.1200379425410095e-6, 2.805632600815787e-5, 3.759464715100376e-5, 8.84115216688531e-5],
    linf = [5.934112354222254e-5, 0.00075475390405777, 0.0008162778009123128, 0.002206991473730824],
    volume_flux = flux_shima_etal)
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_ranocha" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
    l2   = [2.120037931908414e-6, 2.805632845562748e-5, 3.759465243706522e-5, 8.841157002762106e-5],
    linf = [5.934036929955422e-5, 0.0007547536380712039, 0.000816277844819191, 0.0022070017103743567],
    volume_flux = flux_ranocha)
  end

  @testset "elixir_euler_vortex_shockcapturing.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
    l2   = [3.8034711509468997e-6, 5.561030973129845e-5, 5.563956603258559e-5, 0.00015706441614772137],
    linf = [8.493408680687597e-5, 0.0009610606296146518, 0.0009684675522437791, 0.003075812221315033])
  end

  @testset "elixir_euler_vortex_mortar_shockcapturing.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_shockcapturing.jl"),
    l2   = [2.1205855860697905e-6, 2.805356649496243e-5, 3.7617723084029226e-5, 8.841527980901164e-5],
    linf = [5.9005286894620035e-5, 0.0007547295163081724, 0.0008176139355887679, 0.0022089993378280326])
  end

  @testset "elixir_euler_vortex_amr.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
    l2   = [2.0750351586876505e-6, 0.003281637561081054, 0.0032807189382436106, 0.0046470466205649425],
    linf = [4.625172721961501e-5, 0.0318570623352572, 0.031910329823320094, 0.04575283708569344])
  end

  @testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.00011134329189093838, 5.879976755457151e-6, 5.879976755467383e-6, 8.43292512821684e-6, 1.294413664064275e-6, 1.223741200505245e-6, 1.2237412005047353e-6, 1.830592551435133e-6, 8.09851353166173e-7],
      linf = [0.00025633693905779964, 1.637838401076508e-5, 1.6378384010834468e-5, 2.5876901450699874e-5, 5.328007194216333e-6, 8.117611906777178e-6, 8.117611907221267e-6, 1.2107439807826359e-5, 4.179619640695018e-6])
  end

  @testset "elixir_mhd_alfven_wave_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
      l2   = [4.6108068457302814e-6, 1.6898826591958427e-6, 1.62095662606928e-6, 1.6995556746367572e-6, 1.4864336266927356e-6, 1.3876832424912641e-6, 1.3412742751318605e-6, 1.715604818239931e-6, 9.813857517439322e-7],
      linf = [3.52256398420403e-5, 1.535016256673516e-5, 1.4264756196383233e-5, 1.4421626345226257e-5, 7.74417905358149e-6, 1.0188351244111438e-5, 9.862422354345313e-6, 1.6018077102053496e-5, 5.563780231493834e-6],
      tspan = (0.0, 1.0))
  end

  @testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.03607874387815601, 0.04281384915855416, 0.04280200504438979, 0.025747003430056554, 0.16115322058597656, 0.01745612253322887, 0.017456929800049998, 0.02688310667858283, 0.00014430894113852912],
      linf = [0.23502440604318453, 0.3156052602680334, 0.3122357373601531, 0.2117838867907168, 0.9755880857073289, 0.09106404393690148, 0.09184765875826928, 0.15684980169609763, 0.004491681184334955])
  end

  @testset "elixir_mhd_orszag_tang.jl" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
    l2   = [0.2163647978089648, 0.2634048903415803, 0.31384196573588496, 0.0, 0.5118708667236842, 0.22897105894893263, 0.3429130581710548, 0.0, 0.003323963290316924],
    linf = [1.2449183974695166, 0.6654459536984235, 0.8686051282955087, 0.0, 2.758799749621969, 0.6657952127235479, 0.9582341692707345, 0.0, 0.04658967678307848],
    tspan = (0.0, 0.09))
  end

  @testset "elixir_mhd_orszag_tang.jl with flux_hll" begin
  @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
    l2   = [0.10798888204280074, 0.20183486653700633, 0.22972552402685578, 0.0, 0.2994388650416645, 0.15698974423167006, 0.24300134748798047, 0.0, 0.01174975780677972],
    linf = [0.5598143002919851, 0.510328469143869, 0.6563037977791791, 0.0, 0.9813391895361989, 0.3991470482864098, 0.6745300155313261, 0.0, 0.26758015036861943],
    tspan = (0.0, 0.06), surface_flux = flux_hll)
  end
end

# Coverage test for all initial conditions
@testset "Tests for initial conditions" begin
  # TODO Taal: create separate elixirs for ICs/BCs to keep `basic` simple
  # Linear scalar advection
  @testset "elixir_advection_basic.jl with initial_condition_sin_sin" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.0001424424804667062],
      linf = [0.0007260692243250544],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_sin_sin)
  end

  @testset "elixir_advection_basic.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [6.120436421866528e-16],
      linf = [1.3322676295501878e-15],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_advection_basic.jl with initial_condition_linear_x_y" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [2.559042358408011e-16],
      linf = [6.8833827526759706e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x_y,
      boundary_conditions = Trixi.boundary_condition_linear_x_y,
      periodicity=false)
  end

  @testset "elixir_advection_basic.jl with initial_condition_linear_x" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [1.5901063275642836e-16],
      linf = [1.5543122344752192e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x,
      boundary_conditions = Trixi.boundary_condition_linear_x,
      periodicity=false)
  end

  @testset "elixir_advection_basic.jl with initial_condition_linear_y" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [1.597250146891042e-16],
      linf = [3.552713678800501e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_y,
      boundary_conditions = Trixi.boundary_condition_linear_y,
      periodicity=false)
  end

  # Compressible Euler
  @testset "elixir_euler_vortex.jl one step with initial_condition_density_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.003201074851451383, 0.0032010748514513724, 0.0032010748514513716, 0.0032010748514513794],
      linf = [0.043716393835876444, 0.043716393835876444, 0.043716393835876, 0.04371639383587578],
      maxiters = 1,
      initial_condition = initial_condition_density_pulse)
  end

  @testset "elixir_euler_vortex.jl one step with initial_condition_pressure_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.00018950189533270512, 0.0020542290689775757, 0.002054229068977579, 0.01013381064979542],
      linf = [0.004763284475434837, 0.028439617580275578, 0.028439617580275467, 0.13640572175447918],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_pressure_pulse)
  end

  @testset "elixir_euler_vortex.jl one step with initial_condition_density_pressure_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.0031880440066425803, 0.0050397619349217574, 0.005039761934921767, 0.014340770024960708],
      linf = [0.04279723800834989, 0.06783565847184869, 0.06783565847184914, 0.19291274039254347],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_density_pressure_pulse)
  end

  @testset "elixir_euler_vortex.jl one step with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [2.359732835648237e-16, 1.088770274131804e-16, 1.1814939065033234e-16, 1.980283448445849e-15],
      linf = [4.440892098500626e-16, 2.914335439641036e-16, 4.718447854656915e-16, 3.552713678800501e-15],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_euler_sedov_blast_wave.jl one step" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.0021037031798961936, 0.010667428589443041, 0.010667428589443027, 0.11041565217737695],
      linf = [0.11754829172684966, 0.7227194329885249, 0.7227194329885249, 5.42708544137305],
      maxiters=1)
  end

  @testset "elixir_euler_sedov_blast_wave.jl one step with initial_condition_medium_sedov_blast_wave" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.002102553227287478, 0.01066154856802227, 0.010661548568022277, 0.11037470219676422],
      linf = [0.11749257043751615, 0.7223475657303381, 0.7223475657303381, 5.425015419074852],
      maxiters=1, initial_condition=initial_condition_medium_sedov_blast_wave)
  end

  # GLM-MHD
  @testset "elixir_mhd_alfven_wave.jl one step with initial_condition_constant" begin
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
        l2   = [1.9377318494777845e-16, 2.0108417179968547e-16, 4.706803550379074e-16, 9.849916218369067e-17, 9.578096259273606e-15, 4.995499731290712e-16, 2.72017579525395e-16, 9.963303137205655e-17, 1.7656549191657418e-16],
        linf = [4.440892098500626e-16, 7.494005416219807e-16, 1.7763568394002505e-15, 2.220446049250313e-16, 2.1316282072803006e-14, 1.3322676295501878e-15, 8.881784197001252e-16, 2.220446049250313e-16, 7.414582366945819e-16],
        maxiters = 1,
        initial_condition = initial_condition_constant)
  end

  @testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
      l2   = [1.242331325706948, 1.7997067761766712, 1.6900744901036713, 0.0, 2.2633102527850792, 0.2126428351977352, 0.23325824999860192, 0.0, 0.0022978341102783454],
      linf = [10.47234840811493, 14.048907116538157, 15.54763380440781, 0.0, 16.62467416975165, 1.301547732573656, 1.412056956742067, 0.0, 0.05816617137840789],
      tspan = (0.0, 0.05))
  end

  @testset "elixir_mhd_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave.jl"),
      l2   = [0.17583256969707384, 3.85412769737864, 2.4732090396655964, 0.0, 355.0088941187941, 2.343671752007696, 1.3931083969474818, 0.0, 0.022610230392501558],
      linf = [1.5945047955953267, 44.308288031541416, 12.853903528763212, 0.0, 2207.2376410683796, 12.648257494989505, 9.014735778889115, 0.0, 0.3410232617564298],
      tspan = (0.0, 0.003))
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

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D

end #module
