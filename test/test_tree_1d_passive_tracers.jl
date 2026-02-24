module TestExamples1DEuler

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")

@testset "Passive Tracers Tree 1D" begin
#! format: noindent

@trixi_testset "elixir_euler_density_wave_tracers.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave_tracers.jl"),
                        l2=[0.07817688029733633,
                            0.007817688029733637,
                            0.0003908844014910887,
                            0.11826401699443158,
                            0.09888629862239204],
                        linf=[0.23661504279664292,
                            0.023661504279667844,
                            0.0011830752140795653,
                            0.2751965175624824,
                            0.16065446067022204])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end # testset
end # module
