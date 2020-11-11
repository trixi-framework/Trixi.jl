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
  # Linear advection
  @testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @testset "elixir_advection_basic.jl with polydeg=1" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [0.052738240907090894],
      linf = [0.08754218386076529],
      polydeg=1)
  end

  @testset "elixir_advection_timeintegration.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [9.144681778837444e-6],
      linf = [6.437440532436334e-5])
  end

  @testset "elixir_advection_timeintegration.jl with carpenter_kennedy_erk43" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [8.90896259060052e-6],
      linf = [6.969419032576418e-5],
      ode_algorithm=Trixi.CarpenterKennedy2N43(),
      cfl = 1.0)
  end

  @testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk94" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [7.932405176905278e-6],
      linf = [6.509399993848142e-5],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar94())
  end

  @testset "elixir_advection_timeintegration.jl with parsani_ketcheson_deconinck_erk32" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_timeintegration.jl"),
      l2   = [0.004405427606459577],
      linf = [0.012549162970726613],
      ode_algorithm=Trixi.ParsaniKetchesonDeconinck3Sstar32())
  end

  @testset "elixir_advection_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
      l2   = [0.0016133676309497648],
      linf = [0.013960195840607481])
  end

  @testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      l2   = [0.009907208939244499],
      linf = [0.04335954152178878])
  end

  @testset "elixir_advection_amr_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
      l2   = [0.007013561257721758],
      linf = [0.039176916074623536])
  end

  @testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [1.2148032454832534e-5],
      linf = [6.495644795245781e-5])
  end

  @testset "elixir_advection_callbacks.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_callbacks.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end


  # Hyperbolic diffusion
  @testset "elixir_hypdiff_lax_friedrichs.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
      l2   = [0.00015687751817403066, 0.001025986772216324, 0.0010259867722164071],
      linf = [0.001198695637957381, 0.006423873515531753, 0.006423873515533529])
  end

  @testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_harmonic_nonperiodic.jl"),
      l2   = [8.618132355121019e-8, 5.619399844384306e-7, 5.619399844844044e-7],
      linf = [1.1248618588430072e-6, 8.622436487026874e-6, 8.622436487915053e-6])
  end

  @testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
      l2   = [8.523077653954864e-6, 2.8779323653020624e-5, 5.454942769125663e-5],
      linf = [5.522740952468297e-5, 0.00014544895978971679, 0.00032396328684924924])
  end

  @testset "elixir_hypdiff_upwind.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_upwind.jl"),
      l2   = [5.868147556427088e-6, 3.80517927324465e-5, 3.805179273249344e-5],
      linf = [3.701965498725812e-5, 0.0002122422943138247, 0.00021224229431116015])
  end


  # Compressible Euler
  @testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5])
  end

  @testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.0010600778457964775, 0.00010600778457634275, 0.00021201556915872665, 2.650194614399671e-5],
      linf = [0.006614198043413566, 0.0006614198043973507, 0.001322839608837334, 0.000165354951256802],
      tspan = (0.0, 0.5))
  end

  @testset "elixir_euler_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic.jl"),
      l2   = [2.3653424742011065e-6, 2.1388875092652046e-6, 2.1388875093792834e-6, 6.010896863407163e-6],
      linf = [1.4080465934984687e-5, 1.7579850582816192e-5, 1.757985059525069e-5, 5.95689353266593e-5])
  end

  @testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.06172864640680411, 0.050194807377561185, 0.050202324800403486, 0.22588683333743503],
      linf = [0.29813572480585526, 0.3069377110825767, 0.306807092333435, 1.062952871675828])
  end

  @testset "elixir_euler_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [0.13938437665766198, 0.11552277347912725, 0.11552281282617058, 0.3391365873985103],
      linf = [1.4554615795909327, 1.3256380647458283, 1.3256380647457329, 1.8161954000082436],
      maxiters = 30)
  end

  @testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.053797946432602085, 0.04696120828935379, 0.04696384063506395, 0.19685320969570913],
      linf = [0.18540158860112732, 0.24029373364236004, 0.23267525584314722, 0.6874555954921888])
  end

  @testset "elixir_euler_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
      l2   = [0.6777595378750056, 0.2813201698283611, 0.2813199969265205, 0.720079968985097],
      linf = [2.889020548335417, 1.8035429384705923, 1.8034830438685023, 3.033907232631001],
      tspan = (0.0, 1.0))
  end

  @testset "elixir_euler_sedov_blast_wave.jl with tend = 1.0" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.4819231474652106, 0.1655453843075888, 0.16554538430758944, 0.6182451347074112],
      linf = [2.4849541392059202, 1.2816207838889486, 1.2816207838888944, 6.4743208586529875],
      tspan = (0.0, 1.0))
  end

  @testset "elixir_euler_blob_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_mortar.jl"),
      l2   = [0.22112279110928998, 0.6274680581327996, 0.24324346433572555, 2.925884837102588],
      linf = [10.209709958634505, 28.427728124038527, 9.45255862469211, 97.24756873689934],
      tspan = (0.0, 0.5))
  end

  @testset "elixir_euler_blob_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_amr.jl"),
      l2   = [0.20227473253519843, 1.1852146938118358, 0.10163211952508579, 5.24084854294007],
      linf = [14.241296885123061, 71.64475965226391, 7.3265332346563605, 293.17007294795917],
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
        l2   = [0.002046113073936985, 0.002862623943300569, 0.001971116879236713, 0.004816623657677065],
        linf = [0.024375050653856478, 0.01803061241763637, 0.009938942915093363, 0.02097211774984231],
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
        l2   = [0.0016547879036312315, 0.0023815990489650233, 0.0013738124611069249, 0.0031578210702852003],
        linf = [0.022466749278200915, 0.016671527732386116, 0.007178033902807723, 0.015014697702609325],
        tspan = (0.0, 0.2))
    else
      @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability_amr.jl"),
        tspan = (0.0, 0.2))
    end
  end

  @testset "elixir_euler_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [3.6343141788303172e-6, 0.0032111379945554378, 0.0032111482803927763, 0.004545715899533244],
      linf = [7.901851921288117e-5, 0.030561510550305093, 0.03050266851613559, 0.042876690674344076])
  end

  @testset "elixir_euler_vortex_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
      l2   = [2.1203067597905396e-6, 2.792922844344066e-5, 3.7593422075434476e-5, 8.813644239472934e-5],
      linf = [5.932046017209647e-5, 0.0007491265252463908, 0.0008165690091972433, 0.0022122638482677814])
  end

  @testset "elixir_euler_vortex_mortar_split.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
      l2   = [2.1203686464224497e-6, 2.8053511778944724e-5, 3.7617944213324846e-5, 8.840785095671702e-5],
      linf = [5.900566824046383e-5, 0.0007554116580206216, 0.0008166047918783947, 0.0022090163480150693])
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_central" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
      l2   = [2.1203067597988456e-6, 2.7929228443420677e-5, 3.7593422075440256e-5, 8.813644239501579e-5],
      linf = [5.932046017187442e-5, 0.0007491265252450585, 0.0008165690091986866, 0.0022122638482677814],
      volume_flux = flux_central)
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_shima_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
      l2   = [2.120102590290442e-6, 2.8056524989269332e-5, 3.7595003938956103e-5, 8.84137216636431e-5],
      linf = [5.9341032821902395e-5, 0.000755231667019185, 0.0008152449093754566, 0.0022069874183223703],
      volume_flux = flux_shima_etal)
  end

  @testset "elixir_euler_vortex_mortar_split.jl with flux_ranocha" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
      l2   = [2.1201025794676757e-6, 2.805652743675726e-5, 3.759500922486255e-5, 8.841377002453846e-5],
      linf = [5.93402785843411e-5, 0.0007552314167595942, 0.0008152450162266511, 0.002206997654994325],
      volume_flux = flux_ranocha)
  end

  @testset "elixir_euler_vortex_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
      l2   = [3.803468104603101e-6, 5.561125881446469e-5, 5.564051742708959e-5, 0.00015706487208149976],
      linf = [8.49343523420254e-5, 0.0009602338002179245, 0.0009669941278601657, 0.0030758222779923017])
  end

  @testset "elixir_euler_vortex_mortar_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_shockcapturing.jl"),
      l2   = [2.120587333728832e-6, 2.8053708180265217e-5, 3.7618048362912055e-5, 8.841529146530808e-5],
      linf = [5.900528113544912e-5, 0.0007554176274675584, 0.000816603085540657, 0.002208999078852969])
  end

  @testset "elixir_euler_vortex_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
      l2   = [2.1325938318497577e-6, 0.003281545066125044, 0.0032806321457482615, 0.00464605975938406],
      linf = [4.667437348149228e-5, 0.03175420871507906, 0.0318039789241531, 0.04561735256198318])
  end


  # MHD
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
      l2   = [0.0001424424872539405],
      linf = [0.0007260692243253875],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_sin_sin)
  end

  @testset "elixir_advection_basic.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [3.2933000250376106e-16],
      linf = [6.661338147750939e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_advection_basic.jl with initial_condition_linear_x_y" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [2.478798286796091e-16],
      linf = [7.105427357601002e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x_y,
      boundary_conditions = Trixi.boundary_condition_linear_x_y,
      periodicity=false)
  end

  @testset "elixir_advection_basic.jl with initial_condition_linear_x" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [1.475643203742897e-16],
      linf = [1.5543122344752192e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x,
      boundary_conditions = Trixi.boundary_condition_linear_x,
      periodicity=false)
  end

  @testset "elixir_advection_basic.jl with initial_condition_linear_y" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [1.5465148503676022e-16],
      linf = [3.6637359812630166e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_y,
      boundary_conditions = Trixi.boundary_condition_linear_y,
      periodicity=false)
  end


  # Compressible Euler
  @testset "elixir_euler_vortex.jl one step with initial_condition_density_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.003489659044164644, 0.0034896590441646494, 0.0034896590441646502, 0.003489659044164646],
      linf = [0.04761180654650543, 0.04761180654650565, 0.047611806546505875, 0.04761180654650454],
      maxiters = 1,
      initial_condition = initial_condition_density_pulse)
  end

  @testset "elixir_euler_vortex.jl one step with initial_condition_pressure_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.00021747693728234874, 0.0022010142997830533, 0.0022010142997830485, 0.010855273768135729],
      linf = [0.005451116856088789, 0.03126448432601536, 0.03126448432601536, 0.14844305553724624],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_pressure_pulse)
  end

  @testset "elixir_euler_vortex.jl one step with initial_condition_density_pressure_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.003473649182284682, 0.005490887132955628, 0.005490887132955635, 0.015625074774949926],
      linf = [0.046582178207169145, 0.07332265196082899, 0.07332265196082921, 0.2107979471941368],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_density_pressure_pulse)
  end

  @testset "elixir_euler_vortex.jl one step with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [7.89034964747135e-17, 8.095575651413758e-17, 1.0847287658433571e-16, 1.2897732640029767e-15],
      linf = [2.220446049250313e-16, 3.191891195797325e-16, 4.163336342344337e-16, 3.552713678800501e-15],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_euler_sedov_blast_wave.jl one step" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.0021037031798961914, 0.010667428589443025, 0.01066742858944302, 0.10738893384136498],
      linf = [0.11854059147646778, 0.7407961272348982, 0.7407961272348981, 3.92623931433345],
      maxiters=1)
  end

  @testset "elixir_euler_sedov_blast_wave.jl one step with initial_condition_medium_sedov_blast_wave" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.0021025532272874827, 0.010661548568022292, 0.010661548568022284, 0.10734939168392313],
      linf = [0.11848345578926645, 0.7404217490990809, 0.7404217490990809, 3.9247328712525973],
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
