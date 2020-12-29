module TestExamples3D

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "3d")

# Start with a clean environment: remove Trixi output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive=true)

@testset "3D" begin

# Run basic tests
@testset "Examples 3D" begin
  # Linear scalar advection
  include("test_examples_3d_advection.jl")

  # Hyperbolic diffusion
  include("test_examples_3d_hypdiff.jl")

  # Compressible Euler
  include("test_examples_3d_euler.jl")

  # MHD
  include("test_examples_3d_mhd.jl")

  # Compressible Euler with self-gravity
  include("test_examples_3d_eulergravity.jl")

  # Lattice-Boltzmann
  include("test_examples_3d_lbm.jl")
end


@testset "Displaying components 3D" begin
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
end


@testset "Additional tests in 3D" begin
  @testset "compressible Euler" begin
    eqn = CompressibleEulerEquations3D(1.4)

    @test isapprox(energy_total([1.0, 2.0, 3.0, 4.0, 20.0], eqn), 20.0)
    @test isapprox(energy_kinetic([1.0, 2.0, 3.0, 4.0, 20], eqn), 14.5)
    @test isapprox(energy_internal([1.0, 2.0, 3.0, 4.0, 20], eqn), 5.5)
  end

  @testset "hyperbolic diffusion" begin
    @test_nowarn_debug HyperbolicDiffusionEquations3D(nu=1.0)
    eqn = HyperbolicDiffusionEquations3D(nu=1.0)

    @test isapprox(initial_condition_sedov_self_gravity(collect(1:3), 4.5, eqn), zeros(4))
    @test isapprox(boundary_condition_sedov_self_gravity(collect(1:4), 1, 1, collect(11:13), 2.3, flux_central, eqn), [-1.0, -19.739208802178712, 0.0, 0.0])
    @test isapprox(boundary_condition_sedov_self_gravity(collect(1:4), 2, 2, collect(11:13), 4.5, flux_central, eqn), [-1.5, 0.0, -19.739208802178712, 0.0])
  end

  @testset "ideal GLM MHD" begin
    eqn = IdealGlmMhdEquations3D(1.4)
    u = [1.0, 2.0, 3.0, 4.0, 20.0, 0.1, 0.2, 0.3, 1.5]

    @test isapprox(density(u, eqn), 1.0)
    @test isapprox(pressure(u, eqn), 1.7219999999999995)
    @test isapprox(density_pressure(u, eqn), 1.7219999999999995)

    @test isapprox(Trixi.entropy_thermodynamic(u, eqn), 0.5434864060055388)
    @test isapprox(Trixi.entropy_math(u, eqn), -1.3587160150138473)
    @test isapprox(Trixi.entropy(u, eqn), -1.3587160150138473)

    @test isapprox(energy_total(u, eqn), 20.0)
    @test isapprox(energy_kinetic(u, eqn), 14.5)
    @test isapprox(energy_magnetic(u, eqn), 0.07)
    @test isapprox(energy_internal(u, eqn), 4.305)

    @test isapprox(cross_helicity(u, eqn), 2.0)
  end
end


# Clean up afterwards: delete Trixi output directory
@test_nowarn_debug rm(outdir, recursive=true)

end # 3D

end #module
