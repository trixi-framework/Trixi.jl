module TestTree1DBlockFV

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_blockfv")

@testset "BlockFV 1D" begin
#! format: noindent

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.0006893739166730614],
                        linf=[0.0009749048888131329],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_advection_basic.jl with less n_nodes and higher refinement" begin
    # Compute with more volumes per macro cell.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=8,
                        initial_refinement_level=5,
                        tspan=(0.0, 0.5))
    res1 = analysis_callback(sol)

    # Compute with fewer volumes per macro cell.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=4,
                        initial_refinement_level=6,
                        tspan=(0.0, 0.5))
    res2 = analysis_callback(sol)

    # Both setups have exactly the same degrees of freedom.
    # Thus, they should return the same errors (up to floating-point precision).
    @test res1.l2 ≈ res2.l2
    @test res1.linf ≈ res2.linf

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_advection_basic.jl with same number of DOFs as tree_1d_dgsem/elixir_advection_finite_volume.jl" begin
    # Compute with more volumes per macro cell.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        n_nodes=4,
                        initial_refinement_level=3)
    res1 = @inferred analysis_callback(sol)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)

    # Compute with one volume per macro cell, DG with polydeg = 0.
    @test_trixi_include(joinpath(examples_dir(), "tree_1d_dgsem",
                                 "elixir_advection_finite_volume.jl"),
                        polydeg=0,
                        initial_refinement_level=5)
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

@trixi_testset "elixir_euler_source_term_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_term_nonperiodic.jl"),
                        l2=[
                            0.004626422103035306,
                            0.008180600152697231,
                            0.017245734957489538
                        ],
                        linf=[
                            0.007560269781625273,
                            0.014685499562911097,
                            0.030302774363829776
                        ],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testset "UniformFiniteVolumeBasis and VolumeIntegralFiniteVolume" begin
    basis = UniformFiniteVolumeBasis(4)
    @test Trixi.polydeg(basis) == 0

    integral = VolumeIntegralFiniteVolume(flux_lax_friedrichs)
    @test_nowarn show(IOContext(IOBuffer(), :compact => true), MIME"text/plain"(),
                      integral)
    @test_nowarn show(IOContext(IOBuffer(), :compact => false), MIME"text/plain"(),
                      integral)
end
end # BlockFV 1D

end # module
