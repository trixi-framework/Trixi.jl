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
      l2   = [0.00013446460962856976],
      linf = [0.0012577781391462928])
  end

  @trixi_testset "elixir_advection_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream.jl"),
      l2   = [1.830875777528287e-14],
      linf = [7.491784970170556e-13],
      atol = 8e-13, # required to make tests pass on Windows
      )
  end

  @trixi_testset "elixir_advection_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
      l2   = [6.522004549411137e-5],
      linf = [0.005554857853361295])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [0.0281388160824776],
      linf = [0.08740635193023694])
  end

  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [0.010385936842221583, 0.00977604883389343, 0.009776048833893207, 0.009776048833893053, 0.015066870974161749],
      linf = [0.03285848350791598, 0.03217923164091063, 0.03217923164089953, 0.03217923164090308, 0.06554080233334769])
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

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [0.0015695663270396526, 0.0015490919943869574, 0.0015490919943870625, 0.001549091994386903, 0.0030142321185969594],
      linf = [0.011169568009158803, 0.012122645263176413, 0.012122645263191512, 0.012122645263152654, 0.022766806484107338])
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.011367083018614027, 0.007022020327490176, 0.006759580335962235, 0.006820337637760632, 0.02912659127566544],
      linf = [0.2761764220925329, 0.20286331858055706, 0.18763944865434593, 0.19313636558790004, 0.707563913727584],
      tspan = (0.0, 0.25))
  end

  @trixi_testset "elixir_euler_cubed_sphere_coupled.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_cubed_sphere_coupled.jl"),
    l2   = [0.004431408578259766, 0.004171526709906767, 0.004171526709906793, 0.004171526709906778, 0.008005125796951692], 
    linf = [0.030382705156523082, 0.025131165200025674, 0.04078360878658249, 0.04078360878657472, 0.0774158720823146])
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
