module TestExamples1DBurgers

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Inviscid Burgers" begin
#! format: noindent

@trixi_testset "elixir_burgers_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_basic.jl"),
                        l2=[2.967470209082194e-5],
                        linf=[0.00016152468882624227])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_burgers_linear_stability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_linear_stability.jl"),
                        l2=[0.5660569881106876],
                        linf=[1.9352238038313998])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_burgers_shock.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_shock.jl"),
                        l2=[0.4422505602587537],
                        linf=[1.0000000000000009])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_burgers_rarefaction.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_burgers_rarefaction.jl"),
                        l2=[0.4038224690923722],
                        linf=[1.0049201454652736])
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
