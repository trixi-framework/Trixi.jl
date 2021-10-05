module TestExamples3DStructured

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "structured_3d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "StructuredMesh3D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [0.00016263963870641478], 
      linf = [0.0014537194925779984])
  end

  @trixi_testset "elixir_advection_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream.jl"),
      l2   = [1.2908196366970896e-14], 
      linf = [1.0262901639634947e-12],
      atol = 8e-13, # required to make tests pass on Windows
      )
  end

  @trixi_testset "elixir_advection_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic_curved.jl"),
      l2   = [0.0004483892474201268],
      linf = [0.009201820593762955])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [0.0025903889347585777], 
      linf = [0.018407576968841655])
  end

  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [0.010385936842224346, 0.009776048833895767, 0.00977604883389591, 0.009776048833895733, 0.01506687097416608],
      linf = [0.03285848350791731, 0.0321792316408982, 0.032179231640894645, 0.032179231640895534, 0.0655408023333299])
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [2.8815700334367128e-15, 9.361915278236651e-15, 9.95614203619935e-15, 1.6809941842374106e-14, 1.4815037041566735e-14],
      linf = [4.1300296516055823e-14, 2.0444756998472258e-13, 1.0133560657266116e-13, 2.0627943797535409e-13, 2.8954616482224083e-13])
  end

  @trixi_testset "elixir_euler_free_stream.jl with FluxRotated(flux_lax_friedrichs)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      surface_flux=FluxRotated(flux_lax_friedrichs),
      l2   = [2.8815700334367128e-15, 9.361915278236651e-15, 9.95614203619935e-15, 1.6809941842374106e-14, 1.4815037041566735e-14],
      linf = [4.1300296516055823e-14, 2.0444756998472258e-13, 1.0133560657266116e-13, 2.0627943797535409e-13, 2.8954616482224083e-13])
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic_curved.jl"),
      l2   = [0.0032940531178824463, 0.003275679548217804, 0.0030020672748714084, 0.00324007343451744, 0.005721986362580164],
      linf = [0.03156756290660656, 0.033597629023726316, 0.02095783702361409, 0.03353574465232212, 0.05873635745032857])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.011367083018614027, 0.007022020327490176, 0.006759580335962235, 0.006820337637760632, 0.02912659127566544],
      linf = [0.2761764220925329, 0.20286331858055706, 0.18763944865434593, 0.19313636558790004, 0.707563913727584],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_euler_source_terms_cubed_sphere_coupled.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_cubed_sphere_coupled.jl"),
    l2   = [0.004431408578259766, 0.004171526709906767, 0.004171526709906793, 0.004171526709906778, 0.008005125796951692], 
    linf = [0.030382705156523082, 0.04078360878657472, 0.025131165200025674, 0.04078360878658249, 0.0774158720823146])
  end
  
  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [5.30310390e-02, 2.53167260e-02, 2.64276438e-02, 2.52195992e-02, 3.56830295e-01],
      linf = [6.16356950e-01, 2.50600049e-01, 2.74796377e-01, 2.46448217e-01, 4.77888479e+00],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
      l2   = [0.009082355120108163, 0.00712836187575154, 0.006970331624968278, 0.0068988518494490746,
              0.033020095812269036, 0.0032033898339909085, 0.0030774993838538736, 0.003074001381213843,
              4.192130658631602e-5],
      linf = [0.2883946030582689, 0.25956437344015054, 0.2614364943543665, 0.24617277938134657,
              1.1370443512475847, 0.1278041831463388, 0.13347391885068594, 0.1457563463643099,
              0.0021174246048172563],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.003015476175153681, 0.00145499403283373, 0.0009125744757935803, 0.0017703080480578979,
              0.0013046447673965966, 0.0014564863387645508, 0.0013332311430907598, 0.001647832598455728,
              0.0013647609788548722],
      linf = [0.027510637768610846, 0.02797062834945721, 0.01274249949295704, 0.038940694415543736,
              0.02200825678588325, 0.03167600959583505, 0.021420957993862344, 0.03386589835999665,
              0.01888303191983353])
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl with flux_lax_friedrichs" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [0.003047854479955232, 0.0014572199588782184, 0.0009093737183251411, 0.0017937548694553895,
              0.0013010437110755424, 0.0014545607744895874, 0.001328514015121245, 0.001671342529206066,
              0.0013653963058149186],
      linf = [0.027719103797310463, 0.027570111789910784, 0.012561901006903103, 0.03903568568480584,
              0.021311996934554767, 0.03154849824135775, 0.020996033645485412, 0.03403185137382961,
              0.019488952445771597],
      surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # module
