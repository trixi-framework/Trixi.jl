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


  # Burgers
  include("test_examples_1d_burgers.jl")


  # Hyperbolic diffusion
  include("test_examples_1d_hypdiff.jl")


  # Compressible Euler
  include("test_examples_1d_euler.jl")


  # Compressible Euler Multicomponent
  include("test_examples_1d_eulermulti.jl")


  # MHD
  include("test_examples_1d_mhd.jl")


  # MHD Multicomponent
  include("test_examples_1d_mhdmulti.jl")


  # Compressible Euler with self-gravity
  include("test_examples_1d_eulergravity.jl")


  # Curved Mesh
  include("test_examples_1d_curved.jl")
end

# Coverage test for all initial conditions
@testset "Tests for initial conditions" begin
  # Linear scalar advection
  @trixi_testset "elixir_advection_extended.jl with initial_condition_sin" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [0.00017373554109980247],
      linf = [0.0006021275678165239],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_sin)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [2.441369287653687e-16],
      linf = [4.440892098500626e-16],
      maxiters = 1,
      initial_condition = initial_condition_constant)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_linear_x" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [1.9882464973192864e-16],
      linf = [1.4432899320127035e-15],
      maxiters = 1,
      initial_condition = Trixi.initial_condition_linear_x,
      boundary_conditions = Trixi.boundary_condition_linear_x,
      periodicity=false)
  end

  @trixi_testset "elixir_advection_extended.jl with initial_condition_convergence_test" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
      l2   = [6.1803596620800215e-6],
      linf = [2.4858560899509996e-5],
      maxiters = 1,
      initial_condition = initial_condition_convergence_test,
      boundary_conditions = BoundaryConditionDirichlet(initial_condition_convergence_test),
      periodicity=false)
  end
end


@testset "Displaying components 1D" begin
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

  @test_nowarn show(solver.volume_integral); println()
  @test_nowarn println(solver.volume_integral)
  @test_nowarn display(solver.volume_integral)

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

  # Check whether all output is suppressed if the summary, analysis and alive
  # callbacks are set to the TrivialCallback(). Modelled using `@test_nowarn`
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
@test_nowarn rm(outdir, recursive=true)

end # 1D

end #module
