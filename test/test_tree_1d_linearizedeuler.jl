@testsnippet TreeMesh1DLinearizedEuler begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")
end

@testitem "TreeMesh1D LinearizedEuler: elixir_linearizedeuler_convergence.jl" setup=[
    Setup,
    TreeMesh1DLinearizedEuler
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearizedeuler_convergence.jl"),
                        l2=[
                            0.00010894927270421941,
                            0.00014295255695912358,
                            0.00010894927270421941
                        ],
                        linf=[
                            0.0005154647164193893,
                            0.00048457837684242266,
                            0.0005154647164193893
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "TreeMesh1D LinearizedEuler: elixir_linearizedeuler_gauss_wall.jl" setup=[
    Setup,
    TreeMesh1DLinearizedEuler
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_linearizedeuler_gauss_wall.jl"),
                        l2=[0.650082087850354, 0.2913911415488769, 0.650082087850354],
                        linf=[
                            1.9999505145390108,
                            0.9999720404625275,
                            1.9999505145390108
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
