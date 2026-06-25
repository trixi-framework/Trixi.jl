@testsnippet TreeMesh1DEulerGravity begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")
end

@testitem "TreeMesh1D EulerGravity: elixir_eulergravity_convergence.jl" setup=[Setup, TreeMesh1DEulerGravity] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
                        l2=[
                            0.00021708496949694728, 0.0002913795242132917,
                            0.0006112500956552259
                        ],
                        linf=[
                            0.0004977733237385706, 0.0013594226727522418,
                            0.0020418739554664
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
