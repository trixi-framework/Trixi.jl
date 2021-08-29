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

  @trixi_testset "elixir_euler_ec.jl with flux_kennedy_gruber" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.03481471610306124, 0.027694280613944234, 0.027697905866996532, 0.12932052501462554],
      linf = [0.31052098400669004, 0.3481295959664616, 0.34807152194137336, 1.1044947556170719],
      maxiters = 10,
      surface_flux = flux_kennedy_gruber,
      volume_flux = flux_kennedy_gruber)
  end

  @trixi_testset "elixir_euler_ec.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.03481122603050542, 0.027662840593087695, 0.027665658732350273, 0.12927455860656786],
      linf = [0.3110089578739834, 0.34888111987218107, 0.3488278669826813, 1.1056349046774305],
      maxiters = 10,
      surface_flux = flux_chandrashekar,
      volume_flux = flux_chandrashekar)
  end

  @trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
      l2   = [0.05380378209319121, 0.04696482040740498, 0.046967440141927635, 0.19686385916646665],
      linf = [0.1852693811520424, 0.24028641582658528, 0.2326441454434102, 0.6874069967414047])
  end

  @trixi_testset "elixir_euler_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
      l2   = [0.1399704878290245, 0.11580885255188444, 0.115808885313645, 0.3391969791404581],
      linf = [1.4473441276982646, 1.3355435317505784, 1.335543531751034, 1.8203831482774346],
      maxiters = 30)
  end

  @trixi_testset "elixir_euler_blast_wave_neuralnetwork_perssonperaire.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_neuralnetwork_perssonperaire.jl"),
      l2   = [0.470793846459279, 0.20772743006225688, 0.20772649286847356, 0.7011946441199758],
      linf = [1.459980543578883, 0.976674580177999, 0.9759214032734075, 2.960406757135902],
      initial_refinement_level = 4,
      maxiters = 50)
  end

  @trixi_testset "elixir_euler_blast_wave_neuralnetwork_rayhesthaven.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_neuralnetwork_rayhesthaven.jl"),
      l2   = [0.4667685563568254, 0.20611874363505156, 0.20588110990842964, 0.6993986008683889],
      linf = [1.4753582207406488, 0.9904209156187197, 0.9841399353882521, 2.9346283218997806],
      initial_refinement_level = 4,
      maxiters = 50)
  end

  @trixi_testset "elixir_euler_blast_wave_neuralnetwork_rayhesthaven.jl with mortars" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_neuralnetwork_rayhesthaven.jl"),
      l2   = [0.016398689755743465, 0.03074685501652015, 0.03078660152836722, 0.15176702545248866],
      linf = [0.27612812614937343, 0.5598351620373528, 0.5601415330040513, 2.48372589562021],
      refinement_patches=(
          (type="box", coordinates_min=(-0.25, -0.25), coordinates_max=(0.25, 0.25)),
          (type="box", coordinates_min=(-0.125, -0.125), coordinates_max=(0.125, 0.125)),),
      initial_refinement_level = 4,
      maxiters = 5)
  end

  @trixi_testset "elixir_euler_blast_wave_neuralnetwork_cnn.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_neuralnetwork_cnn.jl"),
      l2   = [0.47425390570612547, 0.20990014031106777, 0.2103622732820483, 0.7022475294762982],
      linf = [1.6366874508175182, 0.9734578279971777, 0.9412921347930632, 2.949655726363399],
      initial_refinement_level = 4,
      maxiters = 50)
  end

  @trixi_testset "elixir_euler_blast_wave_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_pure_fv.jl"),
      l2   = [0.39957047631960346, 0.21006912294983154, 0.21006903549932, 0.6280328163981136],
      linf = [2.20417889887697, 1.5487238480003327, 1.5486788679247812, 2.4656795949035857],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
      l2   = [0.6777351803934385, 0.28143110338521693, 0.28143035507881886, 0.7211949078634373],
      linf = [2.881942488461436, 1.8037294505391677, 1.803677581960154, 3.037915614619938],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.4819231474652106, 0.1655453843075888, 0.16554538430758944, 0.6182451347074112],
      linf = [2.4849541392059202, 1.2816207838889486, 1.2816207838888944, 6.4743208586529875],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_sedov_blast_wave_neuralnetwork_perssonperaire.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave_neuralnetwork_perssonperaire.jl"),
      l2   = [8.29516192e-02, 9.18870398e-02, 9.18870398e-02, 4.37449170e-01],
      linf = [1.23402442e+00, 1.63254552e+00, 1.63254552e+00, 4.21977821e+00],
      maxiters = 30)
  end

  @trixi_testset "elixir_euler_positivity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_positivity.jl"),
      l2   = [0.48401054859424775, 0.16613328471131006, 0.16613328471131006, 0.6181793382171535],
      linf = [2.4903327485761544, 1.2898210694161085, 1.2898210694161072, 6.4723873993158385],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_blob_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_mortar.jl"),
      l2   = [0.22206706482833133, 0.6296849068200803, 0.24351238064525166, 2.9118777552679345],
      linf = [9.584083314003589, 27.72635365994101, 9.343044146443756, 101.53747277352629],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_blob_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blob_amr.jl"),
      l2   = [0.20150296718390992, 1.1816675092370863, 0.10129652123574924, 5.230357225596295],
      linf = [14.174958287930844, 71.16876942972455, 7.2799739019809415, 291.67608863630244],
      tspan = (0.0, 0.12))
  end

  @trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability.jl"),
      l2   = [0.055689867638859906, 0.032985369421826734, 0.052243738479375024, 0.08009259073134713],
      linf = [0.24051443859674326, 0.16611572070980668, 0.123559477474734, 0.2694484247345663],
      tspan = (0.0, 0.2))
  end

  @trixi_testset "elixir_euler_kelvin_helmholtz_instability_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability_amr.jl"),
      l2   = [0.05569140599245201, 0.033108757924402134, 0.05223526059785845, 0.08007239985228143],
      linf = [0.2540392117514094, 0.17432468822204936, 0.12323401271048683, 0.26897998287356195],
      tspan = (0.0, 0.2))
  end

  @trixi_testset "elixir_euler_kelvin_helmholtz_instability_amr_neuralnetwork_perssonperaire.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_kelvin_helmholtz_instability_amr_neuralnetwork_perssonperaire.jl"),
      l2   = [0.0009932162527962933, 0.004946122604416242, 0.004863174253792848, 0.005019630958200973],
      linf = [0.008603757244273202, 0.02090493316272357, 0.01712529102069524, 0.028966987799588395],
      maxiters = 30)
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
      l2   = [2.1201025902895562e-6, 2.8056524989293713e-5, 3.759500393896144e-5, 8.84137216637438e-5],
      linf = [5.934103282212444e-5, 0.0007552316670187409, 0.0008152449093751235, 0.0022069874183294758])
  end

  @trixi_testset "elixir_euler_vortex_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
      l2   = [3.7998676433961623e-6, 5.563687203373343e-5, 5.561628017766063e-5, 0.00015707075214260475],
      linf = [8.522682141076654e-5, 0.0009610342345369727, 0.0009656468947373265, 0.0030861481615822584])
  end

  @trixi_testset "elixir_euler_vortex_mortar_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar_shockcapturing.jl"),
      l2   = [2.1203212494737245e-6, 2.8056721454951553e-5, 3.759510818832351e-5, 8.842116438932004e-5],
      linf = [5.9340649220973596e-5, 0.0007552376287922602, 0.0008152432005811283, 0.0022069702233196153])
  end

  @trixi_testset "elixir_euler_vortex_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
      l2   = [2.1325938318497577e-6, 0.003281545066125044, 0.0032806321457482615, 0.00464605975938406],
      linf = [4.667437348149228e-5, 0.03175420871507906, 0.0318039789241531, 0.04561735256198318])
  end
end

# Coverage test for all initial conditions
@testset "Compressible Euler: Tests for initial conditions" begin
  @trixi_testset "elixir_euler_vortex.jl one step with initial_condition_density_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.003489659044164644, 0.0034896590441646494, 0.0034896590441646502, 0.003489659044164646],
      linf = [0.04761180654650543, 0.04761180654650565, 0.047611806546505875, 0.04761180654650454],
      maxiters = 1,
      initial_condition = initial_condition_density_pulse)
  end

  @trixi_testset "elixir_euler_vortex.jl one step with initial_condition_pressure_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.00021747693728234874, 0.0022010142997830533, 0.0022010142997830485, 0.010855273768135729],
      linf = [0.005451116856088789, 0.03126448432601536, 0.03126448432601536, 0.14844305553724624],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_pressure_pulse)
  end

  @trixi_testset "elixir_euler_vortex.jl one step with initial_condition_density_pressure_pulse" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [0.003473649182284682, 0.005490887132955628, 0.005490887132955635, 0.015625074774949926],
      linf = [0.046582178207169145, 0.07332265196082899, 0.07332265196082921, 0.2107979471941368],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_density_pressure_pulse)
  end

  @trixi_testset "elixir_euler_vortex.jl one step with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [7.89034964747135e-17, 8.095575651413758e-17, 1.0847287658433571e-16, 1.2897732640029767e-15],
      linf = [2.220446049250313e-16, 3.191891195797325e-16, 4.163336342344337e-16, 3.552713678800501e-15],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl one step" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.0021037031798961914, 0.010667428589443025, 0.01066742858944302, 0.10738893384136498],
      linf = [0.11854059147646778, 0.7407961272348982, 0.7407961272348981, 3.92623931433345],
      maxiters=1)
  end

  @trixi_testset "elixir_euler_sedov_blast_wave.jl one step with initial_condition_medium_sedov_blast_wave" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
      l2   = [0.0021025532272874827, 0.010661548568022292, 0.010661548568022284, 0.10734939168392313],
      linf = [0.11848345578926645, 0.7404217490990809, 0.7404217490990809, 3.9247328712525973],
      maxiters=1, initial_condition=initial_condition_medium_sedov_blast_wave)
  end
end

end # module
