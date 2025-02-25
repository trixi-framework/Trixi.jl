module TestExamples1DMaxwell

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Maxwell" begin
#! format: noindent

@trixi_testset "elixir_maxwell_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_maxwell_convergence.jl"),
                        l2=[8933.196486422636, 2.979793603210305e-5],
                        linf=[21136.527033627033, 7.050386515528029e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_maxwell_E_excitation.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_maxwell_E_excitation.jl"),
                        l2=[1.8181768208894413e6, 0.09221738723979069],
                        linf=[2.5804473693440557e6, 0.1304024464192847])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end

end # module
