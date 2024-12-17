module TestExamples2DIdealGlmMhdMultiIon

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "tree_2d_dgsem")

@testset "MHD Multi-ion" begin
#! format: noindent

@trixi_testset "elixir_mhdmultiion_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.018116158127836963,
                            0.018242057606900185,
                            0.02842532060540296,
                            0.015805662470907644,
                            0.018125738668209504,
                            0.01841652763250858,
                            0.005540867360495384,
                            0.2091437538608106,
                            0.021619156799420326,
                            0.030561821442897923,
                            0.029778012392807182,
                            0.018651133866567658,
                            0.1204153285953923,
                            0.00017580766470956546
                        ],
                        linf=[
                            0.08706810449620284,
                            0.085845022468381,
                            0.15301576281555795,
                            0.07924698349506726,
                            0.11185098353068998,
                            0.10902813348518392,
                            0.03512713073214368,
                            1.018607282408099,
                            0.11897673517549667,
                            0.18757934071639615,
                            0.19089795413679128,
                            0.10989483685170078,
                            0.6078381519649727,
                            0.00673110606965085
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

@trixi_testset "elixir_mhdmultiion_ec.jl with local Lax-Friedrichs at the surface" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.017668026737187294,
                            0.017797845988889206,
                            0.02784171335711194,
                            0.015603472482130114,
                            0.017849085712788024,
                            0.018142011483140937,
                            0.005478230098832851,
                            0.20585547149049335,
                            0.021301307197244185,
                            0.030185081369414384,
                            0.029385190432285654,
                            0.01837280802521888,
                            0.11810313415151609,
                            0.0002962121353575025
                        ],
                        linf=[
                            0.06594404679380572,
                            0.0658741409805832,
                            0.09451390288403083,
                            0.06787235653416557,
                            0.0891017651119973,
                            0.08828137594061974,
                            0.02364750099643367,
                            0.8059321345611803,
                            0.12243877249999213,
                            0.15930897967901766,
                            0.1538327998380914,
                            0.08694877996306204,
                            0.49493751138636366,
                            0.003287414714660175
                        ],
                        surface_flux=(flux_lax_friedrichs, flux_nonconservative_central))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "Multi-ion GLM-MHD collision source terms" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_collisions.jl"),
                        l2=[
                            0.0,
                            0.0,
                            0.0,
                            1.6829845497998036e-17,
                            0.059553423755556875,
                            5.041277811626498e-19,
                            0.0,
                            0.01971848646448756,
                            7.144325530681256e-17,
                            0.059553423755556924,
                            5.410518863721139e-18,
                            0.0,
                            0.017385071119051767,
                            0.0
                        ],
                        linf=[
                            0.0,
                            0.0,
                            0.0,
                            4.163336342344337e-17,
                            0.05955342375555689,
                            1.609474550734496e-18,
                            0.0,
                            0.019718486464487567,
                            2.220446049250313e-16,
                            0.059553423755556965,
                            1.742701984446815e-17,
                            0.0,
                            0.01738507111905178,
                            0.0
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
