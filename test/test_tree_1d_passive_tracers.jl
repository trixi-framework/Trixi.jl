module TestExamples1DEuler

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Passive Tracers Tree 1D" begin
#! format: noindent

@trixi_testset "elixir_euler_density_wave_tracers.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave_tracers.jl"),
                        l2=[0.08669107435673355,
                            0.004236725669262482,
                            0.057607568740408406,
                            0.12726407019846442,
                            0.1462440384519044],
                        linf=[0.2662334305441576,
                            0.008269815069427132,
                            0.15504600837093108,
                            0.32585227702025543,
                            0.28091670567569604])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end # testset
end # module
