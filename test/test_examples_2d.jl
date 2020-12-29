module TestExamples2D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "2D" begin

# Run basic tests
@testset "Examples 2D" begin
  # Linear advection
  include("test_examples_2d_advection.jl")

  # Hyperbolic diffusion
  include("test_examples_2d_hypdiff.jl")

  # Compressible Euler
  include("test_examples_2d_euler.jl")

  # Compressible Euler Multicomponent
  include("test_examples_2d_eulermulti.jl")

  # MHD
  include("test_examples_2d_mhd.jl")

  # Lattice-Boltzmann
  include("test_examples_2d_lbm.jl")
end

# Coverage test for all initial conditions
@testset "Tests for initial conditions" begin
  # Linear scalar advection
  @testset "elixir_advection_extended.jl with initial_condition_sin_sin" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [0.0001424424872539405],
      linf = [0.0007260692243253875],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_sin_sin)
  end

  @testset "elixir_advection_extended.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [3.2933000250376106e-16],
      linf = [6.661338147750939e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_advection_extended.jl with initial_condition_linear_x_y" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [2.478798286796091e-16],
      linf = [7.105427357601002e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x_y,
      boundary_conditions = Trixi.boundary_condition_linear_x_y,
      periodicity=false)
  end

  @testset "elixir_advection_extended.jl with initial_condition_linear_x" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [1.475643203742897e-16],
      linf = [1.5543122344752192e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x,
      boundary_conditions = Trixi.boundary_condition_linear_x,
      periodicity=false)
  end

  @testset "elixir_advection_extended.jl with initial_condition_linear_y" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
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
      l2   = [7.144325530681224e-17, 2.123397983547417e-16, 5.061138912500049e-16, 3.6588423152083e-17, 8.449816179702522e-15, 3.9171737639099993e-16, 2.445565690318772e-16, 3.6588423152083e-17, 9.971153407737885e-17],
      linf = [2.220446049250313e-16, 8.465450562766819e-16, 1.8318679906315083e-15, 1.1102230246251565e-16, 1.4210854715202004e-14, 8.881784197001252e-16, 4.440892098500626e-16, 1.1102230246251565e-16, 4.779017148551244e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
      l2   = [1.2429643014570362, 1.7996363685163963, 1.6899889382208215, 0.0, 2.263014495582255, 0.2127948919559293, 0.23341167012961367, 0.0, 0.003341061230142962],
      linf = [10.401662016997985, 14.061042946011094, 15.556382737749237, 0.0, 16.714999099689578, 1.3250449624793987, 1.4148467737020096, 0.0, 0.0878998114279951],
      tspan = (0.0, 0.05))
  end

  @testset "elixir_mhd_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave.jl"),
      l2   = [0.17569823349539196, 3.853292514951796, 2.474036084054808, 0.0, 355.36545703316915, 2.3525087027248084, 1.394705056983077, 0.0, 0.029910308624236454],
      linf = [1.5831869462980066, 44.20237543674303, 12.866364462538552, 0.0, 2237.8614138686, 13.066894956593798, 8.984965484247244, 0.0, 0.5226498664960756],
      tspan = (0.0, 0.003))
  end

  # LBM
  @testset "elixir_lbm_couette.jl with initial_condition_couette_steady" begin
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

  @testset "elixir_lbm_lid_driven_cavity.jl with stationary walls" begin
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
