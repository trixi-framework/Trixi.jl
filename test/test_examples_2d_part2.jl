module TestExamples2DPart2

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "2D-Part2" begin

# Run basic tests
@testset "Examples 2D" begin
  # Compressible Euler Multicomponent
  include("test_examples_2d_eulermulti.jl")

  # MHD
  include("test_examples_2d_mhd.jl")

  # MHD Multicomponent
  include("test_examples_2d_mhdmulti.jl")

  # Lattice-Boltzmann
  include("test_examples_2d_lbm.jl")
end

# Coverage test for all initial conditions
@testset "Tests for initial conditions" begin
  # GLM-MHD
  @trixi_testset "elixir_mhd_alfven_wave.jl one step with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
      l2   = [7.144325530681224e-17, 2.123397983547417e-16, 5.061138912500049e-16, 3.6588423152083e-17, 8.449816179702522e-15, 3.9171737639099993e-16, 2.445565690318772e-16, 3.6588423152083e-17, 9.971153407737885e-17],
      linf = [2.220446049250313e-16, 8.465450562766819e-16, 1.8318679906315083e-15, 1.1102230246251565e-16, 1.4210854715202004e-14, 8.881784197001252e-16, 4.440892098500626e-16, 1.1102230246251565e-16, 4.779017148551244e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant,
      atol = 2.0e-13)
  end

  @trixi_testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
      l2   = [1.2429643014570362, 1.7996363685163963, 1.6899889382208215, 0.0, 2.263014495582255, 0.2127948919559293, 0.23341167012961367, 0.0, 0.003341061230142962],
      linf = [10.401662016997985, 14.061042946011094, 15.556382737749237, 0.0, 16.714999099689578, 1.3250449624793987, 1.4148467737020096, 0.0, 0.0878998114279951],
      tspan = (0.0, 0.05))
  end

  @trixi_testset "elixir_mhd_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave.jl"),
      l2   = [0.17569823349539196, 3.853292514951796, 2.474036084054808, 0.0, 355.36545703316915, 2.3525087027248084, 1.394705056983077, 0.0, 0.029910308624236454],
      linf = [1.5831869462980066, 44.20237543674303, 12.866364462538552, 0.0, 2237.8614138686, 13.066894956593798, 8.984965484247244, 0.0, 0.5226498664960756],
      tspan = (0.0, 0.003))
  end

  # LBM
  @trixi_testset "elixir_lbm_couette.jl with initial_condition_couette_steady" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_couette.jl"),
      l2   = [9.321369073400123e-16, 1.6498793963435488e-6, 5.211495843124065e-16,
              1.6520893954826173e-6, 1.0406056181388841e-5, 8.801606429417205e-6,
              8.801710065560555e-6, 1.040614383799995e-5, 2.6135657178357052e-15],
      linf = [1.4432899320127035e-15, 2.1821189867266e-6, 8.881784197001252e-16,
              2.2481261510165496e-6, 1.0692966335143494e-5, 9.606391697600247e-6,
              9.62138334279633e-6, 1.0725969916147021e-5, 3.3861802251067274e-15],
      initial_condition=initial_condition_couette_steady,
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_lbm_lid_driven_cavity.jl with stationary walls" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_lid_driven_cavity.jl"),
      l2   = [1.7198203373689985e-16, 1.685644347036533e-16, 2.1604974801394525e-16,
              2.1527076266915764e-16, 4.2170298143732604e-17, 5.160156233016299e-17,
              6.167794865198169e-17, 5.24166554417795e-17, 6.694740573885739e-16],
      linf = [5.967448757360216e-16, 6.522560269672795e-16, 6.522560269672795e-16,
              6.245004513516506e-16, 2.1163626406917047e-16, 2.185751579730777e-16,
              2.185751579730777e-16, 2.393918396847994e-16, 1.887379141862766e-15],
      boundary_conditions=boundary_condition_wall_noslip,
      tspan = (0, 0.1))
  end
end

# Clean up afterwards: delete Trixi output directory
@test_nowarn rm(outdir, recursive=true)

end # 2D-Part2

end #module
