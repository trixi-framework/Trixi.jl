@testsnippet TreeMesh3DMisc begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_3d_dgsem")
end

@testitem "TreeMesh3D: Additional tests in 3D (compressible Euler)" setup=[Setup] tags=[:tree_part5] begin
    using Trixi: CompressibleEulerEquations3D, energy_total, energy_kinetic,
                 energy_internal
    eqn = CompressibleEulerEquations3D(1.4)

    @test isapprox(energy_total([1.0, 2.0, 3.0, 4.0, 20.0], eqn), 20.0)
    @test isapprox(energy_kinetic([1.0, 2.0, 3.0, 4.0, 20], eqn), 14.5)
    @test isapprox(energy_internal([1.0, 2.0, 3.0, 4.0, 20], eqn), 5.5)
end

@testitem "TreeMesh3D: Additional tests in 3D (hyperbolic diffusion)" setup=[Setup] tags=[:tree_part5] begin
    using Trixi: HyperbolicDiffusionEquations3D
    @test_nowarn HyperbolicDiffusionEquations3D(nu = 1.0)
    eqn = HyperbolicDiffusionEquations3D(nu = 1.0)
end

@testitem "TreeMesh3D: Additional tests in 3D (ideal GLM MHD)" setup=[Setup] tags=[:tree_part6] begin
    using Trixi: Trixi, IdealGlmMhdEquations3D, density, pressure, density_pressure,
                 energy_total, energy_kinetic, energy_magnetic, energy_internal,
                 entropy, entropy_math, entropy_thermodynamic,
                 cross_helicity
    eqn = IdealGlmMhdEquations3D(1.4)
    u = [1.0, 2.0, 3.0, 4.0, 20.0, 0.1, 0.2, 0.3, 1.5]

    @test isapprox(density(u, eqn), 1.0)
    @test isapprox(pressure(u, eqn), 1.7219999999999995)
    @test isapprox(density_pressure(u, eqn), 1.7219999999999995)

    @test isapprox(entropy_thermodynamic(u, eqn), 0.5434864060055388)
    @test isapprox(entropy_math(u, eqn), -1.3587160150138473)
    @test isapprox(entropy(u, eqn), -1.3587160150138473)

    @test isapprox(energy_total(u, eqn), 20.0)
    @test isapprox(energy_kinetic(u, eqn), 14.5)
    @test isapprox(energy_magnetic(u, eqn), 0.07)
    @test isapprox(energy_internal(u, eqn), 4.305)

    @test isapprox(cross_helicity(u, eqn), 2.0)
end

@testitem "TreeMesh3D: Displaying components 3D" setup=[Setup, TreeMesh3DMisc] tags=[:tree_part5] begin
    @test_nowarn include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"))

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
