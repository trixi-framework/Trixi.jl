module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "Polytropic Euler" begin
#! format: noindent

@trixi_testset "elixir_eulerpolytropic_convergence.jl" begin
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end

end # module
