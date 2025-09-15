module TestExamples3DEulerGravity

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_3d_dgsem")

@testset "Compressible Euler with self-gravity" begin
#! format: noindent

@trixi_testset "elixir_eulergravity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
                        l2=[
                            0.0004276779201667428,
                            0.00047204222332596204,
                            0.00047204222332608705,
                            0.0004720422233259819,
                            0.0010987026250960728
                        ],
                        linf=[
                            0.003496616916238704,
                            0.003764418290373106,
                            0.003764418290377103,
                            0.0037644182903766588,
                            0.008370424899251105
                        ],
                        resid_tol=1.0e-4, tspan=(0.0, 0.2))
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
