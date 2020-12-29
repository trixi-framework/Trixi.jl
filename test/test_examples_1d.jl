module TestExamples1D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "1d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "1D" begin

# Run basic tests
@testset "Examples 1D" begin
  # Linear scalar advection
  include("test_examples_1d_advection.jl")


  # Hyperbolic diffusion
  include("test_examples_1d_hypdiff.jl")


  # Compressible Euler
  include("test_examples_1d_euler.jl")


  # MHD
  include("test_examples_1d_mhd.jl")


  # Compressible Euler with self-gravity
  include("test_examples_1d_eulergravity.jl")
end

# Coverage test for all initial conditions
@testset "Tests for initial conditions" begin
  # Linear scalar advection
  @testset "elixir_advection_extended.jl with initial_condition_sin" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [0.00017373554109980247],
      linf = [0.0006021275678165239],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_sin)
  end

  @testset "elixir_advection_extended.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [2.441369287653687e-16],
      linf = [4.440892098500626e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @testset "elixir_advection_extended.jl with initial_condition_linear_x" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [1.9882464973192864e-16],
      linf = [1.4432899320127035e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x,
      boundary_conditions = Trixi.boundary_condition_linear_x,
      periodicity=false)
  end

  @testset "elixir_advection_extended.jl with initial_condition_convergence_test" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [6.1803596620800215e-6],
      linf = [2.4858560899509996e-5],
      maxiters = 1,
      initial_condition = initial_condition_convergence_test,
      boundary_conditions = boundary_condition_convergence_test,
      periodicity=false)
  end
end


@testset "Displaying components 1D" begin
  @test_nowarn_debug include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"))

  # test both short and long printing formats
  @test_nowarn_debug show(mesh); println()
  @test_nowarn_debug println(mesh)
  @test_nowarn_debug display(mesh)

  @test_nowarn_debug show(equations); println()
  @test_nowarn_debug println(equations)
  @test_nowarn_debug display(equations)

  @test_nowarn_debug show(solver); println()
  @test_nowarn_debug println(solver)
  @test_nowarn_debug display(solver)

  @test_nowarn_debug show(solver.basis); println()
  @test_nowarn_debug println(solver.basis)
  @test_nowarn_debug display(solver.basis)

  @test_nowarn_debug show(solver.mortar); println()
  @test_nowarn_debug println(solver.mortar)
  @test_nowarn_debug display(solver.mortar)

  @test_nowarn_debug show(solver.volume_integral); println()
  @test_nowarn_debug println(solver.volume_integral)
  @test_nowarn_debug display(solver.volume_integral)

  @test_nowarn_debug show(semi); println()
  @test_nowarn_debug println(semi)
  @test_nowarn_debug display(semi)

  @test_nowarn_debug show(summary_callback); println()
  @test_nowarn_debug println(summary_callback)
  @test_nowarn_debug display(summary_callback)

  @test_nowarn_debug show(amr_controller); println()
  @test_nowarn_debug println(amr_controller)
  @test_nowarn_debug display(amr_controller)

  @test_nowarn_debug show(amr_callback); println()
  @test_nowarn_debug println(amr_callback)
  @test_nowarn_debug display(amr_callback)

  @test_nowarn_debug show(stepsize_callback); println()
  @test_nowarn_debug println(stepsize_callback)
  @test_nowarn_debug display(stepsize_callback)

  @test_nowarn_debug show(save_solution); println()
  @test_nowarn_debug println(save_solution)
  @test_nowarn_debug display(save_solution)

  @test_nowarn_debug show(analysis_callback); println()
  @test_nowarn_debug println(analysis_callback)
  @test_nowarn_debug display(analysis_callback)

  @test_nowarn_debug show(alive_callback); println()
  @test_nowarn_debug println(alive_callback)
  @test_nowarn_debug display(alive_callback)

  @test_nowarn_debug println(callbacks)

  # Check whether all output is suppressed if the summary, analysis and alive
  # callbacks are set to the TrivialCallback(). Modelled using `@test_nowarn_debug`
  # as basis.
  let fname = tempname()
    try
      open(fname, "w") do f
        redirect_stderr(f) do
          trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        summary_callback=TrivialCallback(),
                        analysis_callback=TrivialCallback(),
                        alive_callback=TrivialCallback())
        end
      end
      @test isempty(read(fname, String))
    finally
      rm(fname, force=true)
    end
  end
end


@testset "Additional tests in 1D" begin
  @testset "compressible Euler" begin
    eqn = CompressibleEulerEquations1D(1.4)

    @test isapprox(Trixi.entropy_thermodynamic([1.0, 2.0, 20.0], eqn), 1.9740810260220094)
    @test isapprox(Trixi.entropy_math([1.0, 2.0, 20.0], eqn), -4.935202565055024)
    @test isapprox(Trixi.entropy([1.0, 2.0, 20.0], eqn), -4.935202565055024)

    @test isapprox(energy_total([1.0, 2.0, 20.0], eqn), 20.0)
    @test isapprox(energy_kinetic([1.0, 2.0, 20.0], eqn), 2.0)
    @test isapprox(energy_internal([1.0, 2.0, 20.0], eqn), 18.0)
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn_debug rm(outdir, recursive=true)

end # 1D

end #module
