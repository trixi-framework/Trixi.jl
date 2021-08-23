module TestExamples2DEuler

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "Compressible Euler" begin
  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [8.517808508019351e-7, 1.2350203856098537e-6, 1.2350203856728076e-6, 4.277886946638239e-6],
      linf = [8.357848139128876e-6, 1.0326302096741458e-5, 1.0326302101404394e-5, 4.496194024383726e-5])
  end

  @trixi_testset "elixir_euler_convergence_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence_pure_fv.jl"),
      l2   = [0.026440292358506527, 0.013245905852168414, 0.013245905852168479, 0.03912520302609374],
      linf = [0.042130817806361964, 0.022685499230187034, 0.022685499230187922, 0.06999771202145322])
  end

  @trixi_testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.0010600778457964775, 0.00010600778457634275, 0.00021201556915872665, 2.650194614399671e-5],
      linf = [0.006614198043413566, 0.0006614198043973507, 0.001322839608837334, 0.000165354951256802],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [2.3653424742011065e-6, 2.1388875092652046e-6, 2.1388875093792834e-6, 6.010896863407163e-6],
      linf = [1.4080465934984687e-5, 1.7579850582816192e-5, 1.757985059525069e-5, 5.95689353266593e-5])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.061751715597716854, 0.05018223615408711, 0.05018989446443463, 0.225871559730513],
      linf = [0.29347582879608825, 0.31081249232844693, 0.3107380389947736, 1.0540358049885143])
  end

  @trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.053797946432602085, 0.04696120828935379, 0.04696384063506395, 0.19685320969570913],
      linf = [0.18540158860112732, 0.24029373364236004, 0.23267525584314722, 0.6874555954921888])
  end

  @trixi_testset "elixir_euler_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [0.13938437665766198, 0.11552277347912725, 0.11552281282617058, 0.3391365873985103],
      linf = [1.4554615795909327, 1.3256380647458283, 1.3256380647457329, 1.8161954000082436],
      maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_nnpp_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [1.39326478e-01, 1.15714111e-01, 1.15714077e-01, 3.39939895e-01],
      linf = [1.44697452e+00, 1.32561295e+00, 1.32561295e+00, 1.81112285e+00],
      maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_nnrh_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [1.37149011e-01, 1.14936250e-01, 1.14814589e-01, 3.39064561e-01],
      linf = [1.47771619e+00, 1.32383912e+00, 1.31919917e+00, 1.81607622e+00],
      maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_cnn_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [1.37604283e-01, 1.14537144e-01, 1.14811941e-01, 3.37168891e-01],
      linf = [1.54890677e+00, 1.39018409e+00, 1.34309015e+00, 1.81474446e+00],
      maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_pure_fv.jl"),
      l2   = [0.39957047631960346, 0.21006912294983154, 0.21006903549932, 0.6280328163981136],
      linf = [2.20417889887697, 1.5487238480003327, 1.5486788679247812, 2.4656795949035857],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
      l2   = [0.6777595378750056, 0.2813201698283611, 0.2813199969265205, 0.720079968985097],
      linf = [2.889020548335417, 1.8035429384705923, 1.8034830438685023, 3.033907232631001],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.4819231474652106, 0.1655453843075888, 0.16554538430758944, 0.6182451347074112],
      linf = [2.4849541392059202, 1.2816207838889486, 1.2816207838888944, 6.4743208586529875],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_positivity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_positivity.jl"),
      l2   = [0.48408821703857413, 0.16610607713651496, 0.16610607713651482, 0.6181627089877778],
      linf = [2.4926053782999267, 1.2903378547503723, 1.2903378547503739, 6.472338570555728],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_blob_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_mortar.jl"),
      l2   = [0.22112279110928998, 0.6274680581327996, 0.24324346433572555, 2.925884837102588],
      linf = [10.209709958634505, 28.427728124038527, 9.45255862469211, 97.24756873689934],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_blob_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_amr.jl"),
      l2   = [0.20227473253519843, 1.1852146938118358, 0.10163211952508579, 5.24084854294007],
      linf = [14.241296885123061, 71.64475965226391, 7.3265332346563605, 293.17007294795917],
      tspan = (0.0, 0.12))
  end

  @trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability.jl"),
      l2   = [5.56898597e-02,   3.29845866e-02,   5.22436730e-02,   8.00923511e-02],
      linf = [2.40499700e-01,   1.66109782e-01,   1.23559478e-01,   2.69558145e-01],
      tspan = (0.0, 0.2))
  end

  @trixi_testset "elixir_euler_kelvin_helmholtz_instability_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability_amr.jl"),
      l2   = [5.56928413e-02,   3.31135409e-02,   5.22350998e-02,   8.00669862e-02],
      linf = [2.53988861e-01,   1.74418201e-01,   1.23234549e-01,   2.69116662e-01],
      tspan = (0.0, 0.2))
  end

  @trixi_testset "elixir_euler_colliding_flow.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_colliding_flow.jl"),
      l2   = [7.23705116e-03,   4.48861301e-02,   8.47273157e-07,   6.62667920e-01],
      linf = [1.93754293e-01,   5.52253475e-01,   4.98055004e-05,   1.50879674e+01],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_euler_colliding_flow_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_colliding_flow_amr.jl"),
      l2   = [6.76960244e-03,   3.21735698e-02,   2.63316434e-07,   6.78416591e-01],
      linf = [2.50119529e-01,   4.07493507e-01,   9.96993324e-05,   2.23204482e+01],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_euler_astro_jet_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_astro_jet_amr.jl"),
      l2   = [0.011338493575859507, 10.097552800226566, 0.0038670992882799088, 4031.828021246965],
      linf = [3.316670548834158, 2992.6286682433883, 7.919515633718346, 1.1914747265545786e6],
      tspan = (0.0, 1.0e-7))
  end

  @trixi_testset "elixir_euler_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [3.6343141788303172e-6, 0.0032111379945554378, 0.0032111482803927763, 0.004545715899533244],
      linf = [7.901851921288117e-5, 0.030561510550305093, 0.03050266851613559, 0.042876690674344076])
  end

  @trixi_testset "elixir_euler_vortex_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
      l2   = [2.1203067597905396e-6, 2.792922844344066e-5, 3.7593422075434476e-5, 8.813644239472934e-5],
      linf = [5.932046017209647e-5, 0.0007491265252463908, 0.0008165690091972433, 0.0022122638482677814])
  end

  @trixi_testset "elixir_euler_vortex_mortar_split.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
      l2   = [2.1203686464224497e-6, 2.8053511778944724e-5, 3.7617944213324846e-5, 8.840785095671702e-5],
      linf = [5.900566824046383e-5, 0.0007554116580206216, 0.0008166047918783947, 0.0022090163480150693])
  end

  @trixi_testset "elixir_euler_vortex_mortar_split.jl with flux_central" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
      l2   = [2.1203067597988456e-6, 2.7929228443420677e-5, 3.7593422075440256e-5, 8.813644239501579e-5],
      linf = [5.932046017187442e-5, 0.0007491265252450585, 0.0008165690091986866, 0.0022122638482677814],
      volume_flux = flux_central)
  end

  @trixi_testset "elixir_euler_vortex_mortar_split.jl with flux_shima_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
      l2   = [2.120102590290442e-6, 2.8056524989269332e-5, 3.7595003938956103e-5, 8.84137216636431e-5],
      linf = [5.9341032821902395e-5, 0.000755231667019185, 0.0008152449093754566, 0.0022069874183223703],
      volume_flux = flux_shima_etal)
  end

  @trixi_testset "elixir_euler_vortex_mortar_split.jl with flux_ranocha" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_split.jl"),
      l2   = [2.1201025794676757e-6, 2.805652743675726e-5, 3.759500922486255e-5, 8.841377002453846e-5],
      linf = [5.93402785843411e-5, 0.0007552314167595942, 0.0008152450162266511, 0.002206997654994325],
      volume_flux = flux_ranocha)
  end

  @trixi_testset "elixir_euler_vortex_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
      l2   = [3.803468104603101e-6, 5.561125881446469e-5, 5.564051742708959e-5, 0.00015706487208149976],
      linf = [8.49343523420254e-5, 0.0009602338002179245, 0.0009669941278601657, 0.0030758222779923017])
  end

  @trixi_testset "elixir_euler_vortex_mortar_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_shockcapturing.jl"),
      l2   = [2.120587333728832e-6, 2.8053708180265217e-5, 3.7618048362912055e-5, 8.841529146530808e-5],
      linf = [5.900528113544912e-5, 0.0007554176274675584, 0.000816603085540657, 0.002208999078852969])
  end

  @trixi_testset "elixir_euler_vortex_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
      l2   = [2.1325938318497577e-6, 0.003281545066125044, 0.0032806321457482615, 0.00464605975938406],
      linf = [4.667437348149228e-5, 0.03175420871507906, 0.0318039789241531, 0.04561735256198318])
  end
end

end # module
