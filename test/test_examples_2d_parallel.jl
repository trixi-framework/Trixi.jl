module TestExamplesParallel2D

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive=true)

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
const EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

@testset "Parallel 2D" begin

# Run basic tests
@testset "Examples 2D" begin
  # Linear scalar advection
  @testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.144681765639205e-6],
      linf = [6.437440532547356e-5])
  end

  @testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [1.2148032444677485e-5],
      linf = [6.495644794757283e-5])
  end

  # Linear scalar advection with AMR
  # These example files are only for testing purposes and have no practical use
  @testset "elixir_advection_amr_refine_twice.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_refine_twice.jl"),
      l2   = [0.017528584408928124],
      linf = [0.06806352260167653])
  end

  @testset "elixir_advection_amr_coarsen_once.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_coarsen_once.jl"),
      l2   = [0.15508361792066527],
      linf = [0.598846070046205])
  end

  # Hyperbolic diffusion
  @testset "elixir_hypdiff_lax_friedrichs.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
      l2   = [0.00015687751816056159, 0.001025986772217084, 0.0010259867722169909],
      linf = [0.0011986956416591976, 0.006423873516411049, 0.006423873516411049])
  end

  @testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_harmonic_nonperiodic.jl"),
      l2   = [8.61813235543625e-8, 5.619399844542781e-7, 5.6193998447443e-7],
      linf = [1.124861862180196e-6, 8.622436471039663e-6, 8.622436470151484e-6])
  end

  @testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
      l2   = [8.523077653955306e-6, 2.8779323653065056e-5, 5.4549427691297846e-5],
      linf = [5.5227409524905013e-5, 0.0001454489597927185, 0.00032396328684569653])
  end

  @testset "elixir_hypdiff_upwind.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_upwind.jl"),
      l2   = [5.868147556385677e-6, 3.805179273239753e-5, 3.805179273248075e-5],
      linf = [3.7019654930525725e-5, 0.00021224229433514097, 0.00021224229433514097])
  end


  # Compressible Euler
  # Note: Some tests here have manually increased relative tolerances since reduction via MPI can
  #       slightly change the L2 error norms (different floating point truncation errors)
  @testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
      l2   = [8.517783186497567e-7, 1.2350199409361865e-6, 1.2350199409828616e-6, 4.277884398786315e-6],
      linf = [8.357934254688004e-6, 1.0326389653148027e-5, 1.0326389654924384e-5, 4.4961900057316484e-5],
      rtol = 2000*sqrt(eps()))
  end

  @testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
      l2   = [0.0010600778457965205, 0.00010600778457646603, 0.0002120155691588112, 2.6501946142012653e-5],
      linf = [0.006614198043407127, 0.0006614198043931596, 0.001322839608845383, 0.00016535495117153687],
      tspan = (0.0, 0.5))
  end

  @testset "elixir_euler_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_nonperiodic.jl"),
      l2   = [2.3652137675654753e-6, 2.1386731303685556e-6, 2.138673130413185e-6, 6.009920290578574e-6],
      linf = [1.4080448659026246e-5, 1.7581818010814487e-5, 1.758181801525538e-5, 5.9568540361709665e-5],
      rtol = 0.001)
  end

  @testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
      l2   = [0.061728646406804005, 0.05019480737756167, 0.050202324800403576, 0.22588683333743628],
      linf = [0.29813572480585526, 0.3069377110825767, 0.306807092333435, 1.062952871675828])
  end

  @testset "elixir_euler_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
      l2   = [3.6342636871275523e-6, 0.0032111366825032443, 0.0032111479254594345, 0.004545714785045611],
      linf = [7.903587114788113e-5, 0.030561314311228993, 0.030502600162385596, 0.042876297246817074],
      rtol = 0.001)
  end

  @testset "elixir_euler_vortex_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_shockcapturing.jl"),
      l2   = [3.80342739421474e-6, 5.561118953968859e-5, 5.564042529709319e-5, 0.0001570628548096201],
      linf = [8.491382365727329e-5, 0.0009602965158113097, 0.0009669978616948516, 0.0030750353269972663],
      rtol = 0.001)
  end
end

# Clean up afterwards: delete Trixi output directory
Trixi.mpi_isroot() && @test_nowarn rm(outdir, recursive=true)

end # Parallel 2D

end # module
