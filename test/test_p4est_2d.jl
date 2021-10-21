module TestExamplesP4estMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_2d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "P4estMesh2D" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as with TreeMesh!
      l2   = [8.311947673061856e-6],
      linf = [6.627000273229378e-5])
  end

  @trixi_testset "elixir_advection_nonconforming_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming_flag.jl"),
      l2   = [3.198940059144588e-5],
      linf = [0.00030636069494005547])
  end

  @trixi_testset "elixir_advection_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_flag.jl"),
      l2   = [0.0005379687442422346],
      linf = [0.007438525029884735])
  end

  @trixi_testset "elixir_advection_amr_solution_independent.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_solution_independent.jl"),
      # Expected errors are exactly the same as with StructuredMesh!
      l2   = [4.949660644033807e-5],
      linf = [0.0004867846262313763])
  end

  @trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_unstructured_flag.jl"),
      l2   = [0.0012766060609964525],
      linf = [0.01750280631586159])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [4.507575525876275e-6],
      linf = [6.21489667023134e-5])
  end

  @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
    l2   = [0.0034516244508588046, 0.0023420334036925493, 0.0024261923964557187, 0.004731710454271893],
    linf = [0.04155789011775046, 0.024772109862748914, 0.03759938693042297, 0.08039824959535657])
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [2.063350241405049e-15, 1.8571016296925367e-14, 3.1769447886391905e-14, 1.4104095258528071e-14],
      linf = [1.9539925233402755e-14, 2e-12, 4.8e-12, 4e-12],
      atol = 2.0e-12, # required to make CI tests pass on macOS
    )
  end

  @trixi_testset "elixir_euler_shockcapturing_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
      l2   = [9.53984675e-02, 1.05633455e-01, 1.05636158e-01, 3.50747237e-01],
      linf = [2.94357464e-01, 4.07893014e-01, 3.97334516e-01, 1.08142520e+00],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [3.76149952e-01, 2.46970327e-01, 2.46970327e-01, 1.28889042e+00],
      linf = [1.22139001e+00, 1.17742626e+00, 1.17742626e+00, 6.20638482e+00],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_euler_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave_amr.jl"),
      l2   = [6.32183914e-01, 3.86914231e-01, 3.86869171e-01, 1.06575688e+00],
      linf = [2.76020890e+00, 2.32659890e+00, 2.32580837e+00, 2.15778188e+00],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_eulergravity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
      l2   = [0.00024871265138964204, 0.0003370077102132591, 0.0003370077102131964, 0.0007231525513793697],
      linf = [0.0015813032944647087, 0.0020494288423820173, 0.0020494288423824614, 0.004793821195083758],
      tspan = (0.0, 0.1))
  end

  @trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [1.0513414461545583e-5, 1.0517900957166411e-6, 1.0517900957304043e-6, 1.511816606372376e-6,
              1.0443997728645063e-6, 7.879639064990798e-7, 7.879639065049896e-7, 1.0628631669056271e-6,
              4.3382328912336153e-7],
      linf = [4.255466285174592e-5, 1.0029706745823264e-5, 1.0029706747467781e-5, 1.2122265939010224e-5,
              5.4791097160444835e-6, 5.18922042269665e-6, 5.189220422141538e-6, 9.552667261422676e-6,
              1.4237578427628152e-6])
  end

  @trixi_testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
      l2   = [0.4552084651735862, 0.8918048264575757, 0.832471223081887, 0.0,
              0.9801264164951583, 0.10475690769435382, 0.1555132409149897, 0.0,
              2.0597079362763556e-5],
      linf = [10.194181233788775, 18.25472397868819, 10.031307436191334, 0.0,
              19.647239392277378, 1.3938810140985936, 1.8724965294853084, 0.0,
              0.0016290067532561904],
      tspan = (0.0, 0.02))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # module
