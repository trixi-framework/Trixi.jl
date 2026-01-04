module TestExamples1DTrafficFlowLWR

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")

@testset "Traffic-flow LWR" begin
#! format: noindent

@trixi_testset "elixir_traffic_flow_lwr_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_traffic_flow_lwr_convergence.jl"),
                        l2=[0.0008455067389588569],
                        linf=[0.004591951086623913])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@trixi_testset "elixir_traffic_flow_lwr_trafficjam.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_traffic_flow_lwr_trafficjam.jl"),
                        l2=[0.1761758135539748], linf=[0.5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
end

end # module
