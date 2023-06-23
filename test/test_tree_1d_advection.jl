module TestExamples1DAdvection

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Linear scalar advection" begin
    @trixi_testset "elixir_advection_basic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            l2=[6.0388296447998465e-6],
                            linf=[3.217887726258972e-5])
    end

    @trixi_testset "elixir_advection_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                            l2=[0.3540206249507417],
                            linf=[0.9999896603382347],
                            coverage_override=(maxiters = 6,))
    end

    @trixi_testset "elixir_advection_amr_nonperiodic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_nonperiodic.jl"),
                            l2=[4.283508859843524e-6],
                            linf=[3.235356127918171e-5],
                            coverage_override=(maxiters = 6,))
    end

    @trixi_testset "elixir_advection_finite_volume.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_finite_volume.jl"),
                            l2=[0.011662300515980219],
                            linf=[0.01647256923710194])
    end
end

end # module
