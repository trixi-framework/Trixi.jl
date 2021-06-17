module TestExamples2DPart1

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "2d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "2D-Part1" begin

# Run basic tests
@testset "Examples 2D" begin
  # Linear advection
  include("test_examples_2d_advection.jl")

  # Hyperbolic diffusion
  include("test_examples_2d_hypdiff.jl")

  # Compressible Euler
  include("test_examples_2d_euler.jl")
end

# Coverage test for all initial conditions
@testset "Tests for initial conditions" begin
  # Linear scalar advection
  @trixi_testset "elixir_advection_extended.jl with initial_condition_sin_sin" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [0.0001424424872539405],
      linf = [0.0007260692243253875],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_sin_sin)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [3.2933000250376106e-16],
      linf = [6.661338147750939e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_linear_x_y" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [2.478798286796091e-16],
      linf = [7.105427357601002e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x_y,
      boundary_conditions = Trixi.boundary_condition_linear_x_y,
      periodicity=false)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_linear_x" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [1.475643203742897e-16],
      linf = [1.5543122344752192e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x,
      boundary_conditions = Trixi.boundary_condition_linear_x,
      periodicity=false)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_linear_y" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [1.5465148503676022e-16],
      linf = [3.6637359812630166e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_y,
      boundary_conditions = Trixi.boundary_condition_linear_y,
      periodicity=false)
  end


  # Compressible Euler
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

end # 2D-Part1

end #module
