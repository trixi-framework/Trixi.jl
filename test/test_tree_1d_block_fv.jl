module TestTree1DBlockFV

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_block_FV")

@testset "BlockFV 1D" begin
#! format: noindent

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_block_FV.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_block_FV.jl"),
                        l2=[0.0006893739166730614],
                        linf=[0.0009749048888131329],
                        tspan=(0.0, 0.5))

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # Linear scalar advection
end # BlockFV 1D

end # module
