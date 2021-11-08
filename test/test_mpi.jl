module TestExamplesMPI

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "MPI 2D" begin

# Run basic tests
@testset "Examples 2D" begin
  # Linear scalar advection
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      # Expected errors are exactly the same as in the serial test!
      l2   = [8.311947673061856e-6],
      linf = [6.627000273229378e-5])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      # Expected errors are exactly the same as in the serial test!
      l2   = [7.81674284320524e-6],
      linf = [6.314906965243505e-5])
  end

  @trixi_testset "elixir_advection_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
      # Expected errors are exactly the same as in the serial test!
      l2   = [0.0015188466707237375],
      linf = [0.008446655719187679])
  end

  @trixi_testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
      # Expected errors are exactly the same as in the serial test!
      l2   = [4.913300828257469e-5],
      linf = [0.00045263895394385967],
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_advection_amr_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
      # Expected errors are exactly the same as in the serial test!
      l2   = [3.2207388565869075e-5],
      linf = [0.0007508059772436404],
      coverage_override = (maxiters=6,))
  end

  # Linear scalar advection with AMR
  # These example files are only for testing purposes and have no practical use
  @trixi_testset "elixir_advection_amr_refine_twice.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_refine_twice.jl"),
      l2   = [0.00020547512522578292],
      linf = [0.007831753383083506],
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_advection_amr_coarsen_twice.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_coarsen_twice.jl"),
      l2   = [0.0014321062757891826],
      linf = [0.0253454486893413],
      coverage_override = (maxiters=6,))
  end

  # Hyperbolic diffusion
  @trixi_testset "elixir_hypdiff_lax_friedrichs.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
      l2   = [0.00015687751816056159, 0.001025986772217084, 0.0010259867722169909],
      linf = [0.0011986956416591976, 0.006423873516411049, 0.006423873516411049])
  end

  @trixi_testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_harmonic_nonperiodic.jl"),
      l2   = [8.61813235543625e-8, 5.619399844542781e-7, 5.6193998447443e-7],
      linf = [1.124861862180196e-6, 8.622436471039663e-6, 8.622436470151484e-6])
  end

  @trixi_testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
      l2   = [8.523077653955306e-6, 2.8779323653065056e-5, 5.4549427691297846e-5],
      linf = [5.5227409524905013e-5, 0.0001454489597927185, 0.00032396328684569653])
  end

  @trixi_testset "elixir_hypdiff_godunov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_godunov.jl"),
      l2   = [5.868147556385677e-6, 3.805179273239753e-5, 3.805179273248075e-5],
      linf = [3.7019654930525725e-5, 0.00021224229433514097, 0.00021224229433514097])
  end


  # Compressible Euler
  # Note: Some tests here have manually increased relative tolerances since reduction via MPI can
  #       slightly change the L2 error norms (different floating point truncation errors)
  @trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [9.321181253186009e-7, 1.4181210743438511e-6, 1.4181210743487851e-6, 4.824553091276693e-6],
      linf = [9.577246529612893e-6, 1.1707525976012434e-5, 1.1707525976456523e-5, 4.8869615580926506e-5],
      rtol = 2000*sqrt(eps()))
  end

  # This example file is only for testing purposes and has no practical use
  @trixi_testset "elixir_euler_source_terms_amr_refine_coarsen.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_amr_refine_coarsen.jl"),
      l2   = [4.8226610349853444e-5, 4.117706709270575e-5, 4.1177067092959676e-5, 0.00012205252427437389],
      linf = [0.0003543874851490436, 0.0002973166773747593, 0.0002973166773760916, 0.001154106793870291],
      # Let this test run until the end to cover the time-dependent lines
      # of the indicator and the MPI-specific AMR code.
      coverage_override = (maxiters=10^5,))
  end

  @trixi_testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.0010600778457965205, 0.00010600778457646603, 0.0002120155691588112, 2.6501946142012653e-5],
      linf = [0.006614198043407127, 0.0006614198043931596, 0.001322839608845383, 0.00016535495117153687],
      tspan = (0.0, 0.5))
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [2.259440511766445e-6, 2.318888155713922e-6, 2.3188881557894307e-6, 6.3327863238858925e-6],
      linf = [1.498738264560373e-5, 1.9182011928187137e-5, 1.918201192685487e-5, 6.0526717141407005e-5],
      rtol = 0.001)
  end

  @trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.061751715597716854, 0.05018223615408711, 0.05018989446443463, 0.225871559730513],
      linf = [0.29347582879608825, 0.31081249232844693, 0.3107380389947736, 1.0540358049885143])
  end

  @trixi_testset "elixir_euler_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [3.53375983916925e-6, 0.0032123259330577325, 0.00321232443824996, 0.004547280616310348],
      linf = [7.719164482999918e-5, 0.030543222729985442, 0.0304822911023237, 0.042888536761282126],
      rtol = 0.001)
  end

  @trixi_testset "elixir_euler_vortex_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
      # Expected errors are exactly the same as in the serial test!
      l2   = [2.110390460364181e-6, 2.7230027429598542e-5, 3.657273339760332e-5, 8.735519818394382e-5],
      linf = [5.9743882399154735e-5, 0.000731856753784843, 0.0007915976735435315, 0.0022215051634404404])
  end

  @trixi_testset "elixir_euler_vortex_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
      # Expected errors are exactly the same as in the serial test!
      l2   = [2.120552206480055e-6, 0.003281541473561042, 0.003280625257336616, 0.004645872821313438],
      linf = [4.500266027052113e-5, 0.031765399304366726, 0.03179340562764421, 0.04563622772500864],
      coverage_override = (maxiters=6,))
  end

  @trixi_testset "elixir_euler_vortex_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
      l2   = [3.7412276700966986e-6, 5.4150680347525463e-5, 5.415287159571093e-5, 0.0001542834620109727],
      linf = [8.473507257800161e-5, 0.0009317864493174621, 0.0009371841830909666, 0.0030735931384739956],
      rtol = 0.001)
  end
end

# Clean up afterwards: delete Trixi output directory
Trixi.mpi_isroot() && @test_nowarn rm(outdir, recursive=true)

end # MPI 2D

end # module
