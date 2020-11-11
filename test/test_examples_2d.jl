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
      l2   = [0.00011134625844259098, 5.880228860293714e-6, 5.8802288602656305e-6, 8.433075065897319e-6, 1.2941683954844375e-6, 1.2241441713310377e-6, 1.2241441713333897e-6, 1.830858349671085e-6, 8.071952648742103e-7],
      linf = [0.0002678884241129609, 1.6247883512621186e-5, 1.6247883513065275e-5, 2.7352876884892408e-5, 5.3276216213093974e-6, 8.117315354660981e-6, 8.117315354883026e-6, 1.2104783261912555e-5, 4.165856899455253e-6])
  end

  @testset "elixir_mhd_alfven_wave_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
      l2   = [4.6082236877167206e-6, 1.6891450234535906e-6, 1.6201997664272474e-6, 1.6994302473445441e-6, 1.4856807642819655e-6, 1.3877537205633802e-6, 1.3411586881667912e-6, 1.715520094466887e-6, 9.799778935649419e-7],
      linf = [3.522195695693231e-5, 1.5344608485887146e-5, 1.4262596865238786e-5, 1.4421435395955973e-5, 7.74339775866384e-6, 1.0192371654671462e-5, 9.862880384736705e-6, 1.601812439049055e-5, 5.563725533915599e-6],
      tspan = (0.0, 1.0))
  end

  @testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.03628125549830369, 0.04300966448828238, 0.04299547820119134, 0.02574623161746654, 0.16204315765266042, 0.017452871572979735, 0.017453938377668458, 0.026879791205306338, 0.00014848784360574238],
      linf = [0.23512983219577177, 0.3156382854505785, 0.30929102216455284, 0.2117216591297295, 0.9716135428189303, 0.0900350025249661, 0.09083429813534472, 0.15755391353234138, 0.003823020513262337])
  end

  @testset "elixir_mhd_orszag_tang.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
      l2   = [0.21666633802441865, 0.26361144322902336, 0.31387224798870406, 0.0, 0.5122169285540558, 0.22914098484830794, 0.343070005986905, 0.0, 0.003135329705500713],
      linf = [1.263856896480327, 0.6746752595848403, 0.8587434327350338, 0.0, 2.8131898016337638, 0.657766103712197, 0.9590640834992721, 0.0, 0.05023931968037013],
      tspan = (0.0, 0.09))
  end

  @testset "elixir_mhd_orszag_tang.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
      l2   = [0.10793922020494437, 0.2017781494189171, 0.22968473063588057, 0.0, 0.29941315651167616, 0.15675853232583936, 0.24281568109143112, 0.0, 0.0036700470485848485],
      linf = [0.5601217528795641, 0.5098906628292882, 0.6567285278153883, 0.0, 0.9899397510955938, 0.3993491794435835, 0.6734632575414139, 0.0, 0.1356178987559583],
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
      l2   = [7.144325530681224e-17, 2.107107684242054e-16, 5.025035342841136e-16, 3.9063728231270855e-17, 8.448122434419674e-15, 3.9171737639099993e-16, 2.404057005740705e-16, 3.6588423152083e-17, 1.609894978924592e-16],
      linf = [2.220446049250313e-16, 8.465450562766819e-16, 1.8318679906315083e-15, 1.1102230246251565e-16, 1.4210854715202004e-14, 8.881784197001252e-16, 4.440892098500626e-16, 1.1102230246251565e-16, 6.384151957784351e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
      l2   = [1.2409483508452785, 1.7949757944358629, 1.6900251218242546, 0.0, 2.2590795557059, 0.2125438074372391, 0.23317061029230146, 0.0, 0.002607027500811714],
      linf = [10.401649981991934, 14.05682271625067, 15.555835126442055, 0.0, 16.727402974668994, 1.3038301364335803, 1.4124333140320517, 0.0, 0.07196402504799167],
      tspan = (0.0, 0.05))
  end

  @testset "elixir_mhd_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave.jl"),
      l2   = [0.1756727350457241, 3.853229982526863, 2.4739762794013913, 0.0, 355.30157063223197, 2.352245499201879, 1.3948411507413514, 0.0, 0.029294695569576597],
      linf = [1.582954476890206, 44.20354450865648, 12.868697011777009, 0.0, 2236.9441159751696, 13.073465853817824, 8.98939612429793, 0.0, 0.5141382637022156],
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
