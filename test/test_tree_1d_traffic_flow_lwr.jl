@testsnippet TreeMesh1DTrafficFlowLWR begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")
end

@testitem "TreeMesh1D Traffic-flow LWR: elixir_traffic_flow_lwr_convergence.jl" setup=[Setup, TreeMesh1DTrafficFlowLWR] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_traffic_flow_lwr_convergence.jl"),
                        l2=[0.0008455067389588569],
                        linf=[0.004591951086623913])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "TreeMesh1D Traffic-flow LWR: elixir_traffic_flow_lwr_trafficjam.jl" setup=[Setup, TreeMesh1DTrafficFlowLWR] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_traffic_flow_lwr_trafficjam.jl"),
                        l2=[0.1761758135539748], linf=[0.5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
