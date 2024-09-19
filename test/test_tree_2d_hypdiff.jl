module TestExamples2DHypDiff

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "Hyperbolic diffusion" begin
#! format: noindent

@trixi_testset "elixir_hypdiff_lax_friedrichs.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_lax_friedrichs.jl"),
                        l2=[
                            0.00015687751817403066,
                            0.001025986772216324,
                            0.0010259867722164071
                        ],
                        linf=[
                            0.001198695637957381,
                            0.006423873515531753,
                            0.006423873515533529
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Larger values for allowed allocations due to usage of custom 
        # integrator which are not *recorded* for the methods from 
        # OrdinaryDiffEq.jl
        # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_hypdiff_harmonic_nonperiodic.jl"),
                        l2=[
                            8.618132355121019e-8,
                            5.619399844384306e-7,
                            5.619399844844044e-7
                        ],
                        linf=[
                            1.1248618588430072e-6,
                            8.622436487026874e-6,
                            8.622436487915053e-6
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

@trixi_testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
                        l2=[
                            8.523077653954864e-6,
                            2.8779323653020624e-5,
                            5.454942769125663e-5
                        ],
                        linf=[
                            5.522740952468297e-5,
                            0.00014544895978971679,
                            0.00032396328684924924
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

@trixi_testset "elixir_hypdiff_godunov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_godunov.jl"),
                        l2=[
                            5.868147556427088e-6,
                            3.80517927324465e-5,
                            3.805179273249344e-5
                        ],
                        linf=[
                            3.701965498725812e-5,
                            0.0002122422943138247,
                            0.00021224229431116015
                        ],
                        atol=2.0e-12) #= required for CI on macOS =#
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
