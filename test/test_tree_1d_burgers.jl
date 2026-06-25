@testsnippet TreeMesh1DBurgers begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_1d_dgsem")
end

@testitem "TreeMesh1D Burgers: elixir_burgers_basic.jl" setup=[Setup, TreeMesh1DBurgers] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_basic.jl"),
                        l2=[2.967470209082194e-5],
                        linf=[0.00016152468882624227])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "TreeMesh1D Burgers: elixir_burgers_linear_stability.jl" setup=[Setup, TreeMesh1DBurgers] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_linear_stability.jl"),
                        l2=[0.5660569881106876],
                        linf=[1.9352238038313998])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "TreeMesh1D Burgers: elixir_burgers_shock.jl" setup=[Setup, TreeMesh1DBurgers] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_shock.jl"),
                        l2=[0.4429871964104191],
                        linf=[1.007778754747701])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end

@testitem "TreeMesh1D Burgers: elixir_burgers_rarefaction.jl" setup=[Setup, TreeMesh1DBurgers] tags=[:tree_part1] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_rarefaction.jl"),
                        l2=[0.4038224690923722],
                        linf=[1.0049201454652736])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs!, semi, sol, 1000)
end
