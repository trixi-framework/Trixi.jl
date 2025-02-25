module TestExamplesTreeMesh3DPart2

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "TreeMesh3D Part 2" begin
#! format: noindent

# Run basic tests
@testset "Examples 3D" begin
    # Linear scalar advection
    include("test_tree_3d_advection.jl")

    # Hyperbolic diffusion
    include("test_tree_3d_hypdiff.jl")

    # Compressible Euler with self-gravity
    include("test_tree_3d_eulergravity.jl")

    # Linearized Euler
    include("test_tree_3d_linearizedeuler.jl")
end

@trixi_testset "Additional tests in 3D" begin
    @trixi_testset "compressible Euler" begin
        eqn = CompressibleEulerEquations3D(1.4)

        @test isapprox(energy_total([1.0, 2.0, 3.0, 4.0, 20.0], eqn), 20.0)
        @test isapprox(energy_kinetic([1.0, 2.0, 3.0, 4.0, 20], eqn), 14.5)
        @test isapprox(energy_internal([1.0, 2.0, 3.0, 4.0, 20], eqn), 5.5)
    end

    @trixi_testset "hyperbolic diffusion" begin
        @test_nowarn HyperbolicDiffusionEquations3D(nu = 1.0)
        eqn = HyperbolicDiffusionEquations3D(nu = 1.0)
    end
end

@trixi_testset "Displaying components 3D" begin
    @test_nowarn include(joinpath(examples_dir(), "tree_3d_dgsem",
                                  "elixir_advection_amr.jl"))

    # test both short and long printing formats
    @test_nowarn show(mesh)
    println()
    @test_nowarn println(mesh)
    @test_nowarn display(mesh)

    @test_nowarn show(equations)
    println()
    @test_nowarn println(equations)
    @test_nowarn display(equations)

    @test_nowarn show(solver)
    println()
    @test_nowarn println(solver)
    @test_nowarn display(solver)

    @test_nowarn show(solver.basis)
    println()
    @test_nowarn println(solver.basis)
    @test_nowarn display(solver.basis)

    @test_nowarn show(solver.mortar)
    println()
    @test_nowarn println(solver.mortar)
    @test_nowarn display(solver.mortar)

    @test_nowarn show(semi)
    println()
    @test_nowarn println(semi)
    @test_nowarn display(semi)

    @test_nowarn show(summary_callback)
    println()
    @test_nowarn println(summary_callback)
    @test_nowarn display(summary_callback)

    @test_nowarn show(amr_controller)
    println()
    @test_nowarn println(amr_controller)
    @test_nowarn display(amr_controller)

    @test_nowarn show(amr_callback)
    println()
    @test_nowarn println(amr_callback)
    @test_nowarn display(amr_callback)

    @test_nowarn show(stepsize_callback)
    println()
    @test_nowarn println(stepsize_callback)
    @test_nowarn display(stepsize_callback)

    @test_nowarn show(save_solution)
    println()
    @test_nowarn println(save_solution)
    @test_nowarn display(save_solution)

    @test_nowarn show(analysis_callback)
    println()
    @test_nowarn println(analysis_callback)
    @test_nowarn display(analysis_callback)

    @test_nowarn show(alive_callback)
    println()
    @test_nowarn println(alive_callback)
    @test_nowarn display(alive_callback)

    @test_nowarn println(callbacks)
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)
end # TreeMesh3D Part 2

end #module
