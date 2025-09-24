module TestExamples1DBurgers

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")

@testset "Linear Elasticity" begin
#! format: noindent

@trixi_testset "elixir_linearelasticity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_linearelasticity_convergence.jl"),
                        l2=[0.0007205516785218745, 0.0008036755866155103],
                        linf=[0.0011507266875070855, 0.003249818227066381])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_linearelasticity_impact.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearelasticity_impact.jl"),
                        l2=[0.004322196488938371, 368483.8160335645],
                        linf=[0.010726138542416276, 999999.9958776952])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end

end # module
