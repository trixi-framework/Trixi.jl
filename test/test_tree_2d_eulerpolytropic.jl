@testsnippet TreeMesh2DEulerPolytropic begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")
end

@testitem "TreeMesh2D Polytropic Euler: elixir_eulerpolytropic_convergence.jl" setup=[
    Setup,
    TreeMesh2DEulerPolytropic
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_eulerpolytropic_convergence.jl"),
                        l2=[
                            0.0016689832177626373, 0.0025920263793094526,
                            0.003281074494626679
                        ],
                        linf=[
                            0.010994883201896677, 0.013309526619350365,
                            0.02008032661117376
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
