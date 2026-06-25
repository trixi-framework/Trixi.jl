module TestTree2DBlockFV

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_blockfv")

@testset "BlockFV 2D" begin
#! format: noindent

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.017295205942012868],
                        linf=[0.02444847499806624],
                        tspan=(0.0, 0.5))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_advection_basic.jl with less n_nodes and higher refinement" begin
    # Compute with more volumes per macro cell.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=4,
                        initial_refinement_level=4,
                        tspan=(0.0, 0.5))
    res1 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Compute with fewer volumes per macro cell.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=2,
                        initial_refinement_level=5,
                        tspan=(0.0, 0.5))
    res2 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Both setups have exactly the same degrees of freedom.
    # Thus, they should return the same errors (up to floating-point precision).
    @test res1.l2 ≈ res2.l2
    @test res1.linf ≈ res2.linf
end
end # Linear scalar advection

@testset "Compressible Euler equations" begin
#! format: noindent

@trixi_testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_density_wave.jl"),
                        l2=[0.031233316749041267,
                            0.003123331674903803,
                            0.006246663349808052,
                            0.0007808329187371395],
                        linf=[0.044169344994492266,
                            0.0044169344994492215,
                            0.008833868998898514,
                            0.0011042336248863194],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_euler_isentropic_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_isentropic_vortex.jl"),
                        l2=[0.0013855544942691467,
                            0.07912269951431652,
                            0.07917097691649295,
                            0.14533624890962035],
                        linf=[0.021277289728345194,
                            0.8434393417500995,
                            0.8127969969547908,
                            2.271903270249524],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_euler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_convergence.jl"),
                        l2=[0.003798391701194144,
                            0.009489467813506548,
                            0.00948946781350655,
                            0.02704154630948781],
                        linf=[0.005743846316061285,
                            0.013649501767585503,
                            0.013649501767585726,
                            0.03876289859195037],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # Compressible Euler equations

@trixi_testset "elixir_euler_vortex_mortar.jl with blockfv vs with dgsem with polydeg=0" begin
    # Compute with blockfv solver.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                        n_nodes=4,
                        initial_refinement_level=5,
                        tspan=(0.0, 0.5))
    res1 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Compute with DGSEM solver with polynomial degree = 0, i.e., a first order finite volume solver.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                        solver=DGSEM(polydeg=0, surface_flux=FluxLaxFriedrichs(max_abs_speed_naive)),
                        initial_refinement_level=7,
                        tspan=(0.0, 0.5))
    res2 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Both setups have exactly the same degrees of freedom.
    # Thus, they should return the same errors (up to floating-point precision).
    @test res1.l2 ≈ res2.l2
    @test res1.linf ≈ res2.linf
end
end # BlockFV 2D

end # module
