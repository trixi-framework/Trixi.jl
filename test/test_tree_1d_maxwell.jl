@testsnippet TreeMesh1DMaxwell begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")
end

@testitem "TreeMesh1D Maxwell: elixir_maxwell_convergence.jl" setup=[
    Setup,
    TreeMesh1DMaxwell
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_maxwell_convergence.jl"),
                        l2=[8933.196486422636, 2.979793603210305e-5],
                        linf=[21136.527033627033, 7.050386515528029e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "TreeMesh1D Maxwell: elixir_maxwell_E_excitation.jl" setup=[
    Setup,
    TreeMesh1DMaxwell
] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_maxwell_E_excitation.jl"),
                        l2=[1.8181768208894413e6, 0.09221738723979069],
                        linf=[2.5804473693440557e6, 0.1304024464192847])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
