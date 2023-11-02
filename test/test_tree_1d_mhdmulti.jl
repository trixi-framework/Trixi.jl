module TestExamples1DMHD

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "MHD Multicomponent" begin
#! format: noindent

@trixi_testset "elixir_mhdmulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
                        l2=[0.08166807325620999, 0.054659228513541616,
                            0.054659228513541616, 0.15578125987042812,
                            4.130462730494e-17, 0.054652588871500665,
                            0.054652588871500665, 0.008307405499637766,
                            0.01661481099927553, 0.03322962199855106],
                        linf=[0.19019207422649645, 0.10059813883022888,
                            0.10059813883022888, 0.4407925743107146,
                            1.1102230246251565e-16, 0.10528911365809623,
                            0.10528911365809623, 0.01737901809766182,
                            0.03475803619532364, 0.06951607239064728])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhdmulti_ec.jl with flux_derigs_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
                        l2=[0.08151404166186461, 0.054640238302693274,
                            0.054640238302693274, 0.15536125426328573,
                            4.130462730494e-17, 0.054665489963920275,
                            0.054665489963920275, 0.008308349501359825,
                            0.01661669900271965, 0.0332333980054393],
                        linf=[0.1824424257860952, 0.09734687137001484,
                            0.09734687137001484, 0.4243089502087325,
                            1.1102230246251565e-16, 0.09558639591092555,
                            0.09558639591092555, 0.017364773041550624,
                            0.03472954608310125, 0.0694590921662025],
                        volume_flux=flux_derigs_etal)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhdmulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_es.jl"),
                        l2=[0.07994082660130175, 0.053940174914031976,
                            0.053940174914031976, 0.15165513559250643,
                            4.130462730494e-17, 0.05363207135290325,
                            0.05363207135290325, 0.008258265884659555,
                            0.01651653176931911, 0.03303306353863822],
                        linf=[0.14101014428198477, 0.07762441749521025,
                            0.07762441749521025, 0.3381334453289866,
                            1.1102230246251565e-16, 0.07003646400675223,
                            0.07003646400675223, 0.014962483760600165,
                            0.02992496752120033, 0.05984993504240066])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhdmulti_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_convergence.jl"),
                        l2=[1.7337265267786785e-5, 0.00032976971029271364,
                            0.0003297697102926479, 6.194071694759044e-5,
                            4.130462730494001e-17, 0.00032596825025664136,
                            0.0003259682502567132, 2.5467510126885455e-5,
                            5.093502025377091e-5, 0.00010187004050754182],
                        linf=[3.877554303711845e-5, 0.0012437848638874956,
                            0.0012437848638876898, 0.00016431262020277781,
                            1.1102230246251565e-16, 0.0012443734922607112,
                            0.001244373492260704, 5.691007974162332e-5,
                            0.00011382015948324664, 0.00022764031896649328])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhdmulti_briowu_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_briowu_shock_tube.jl"),
                        l2=[0.1877830835572639, 0.3455841730726793, 0.0,
                            0.35413123388836687,
                            8.745556626531982e-16, 0.3629920109231055, 0.0,
                            0.05329005553971236,
                            0.10658011107942472],
                        linf=[0.4288187627971754, 1.0386547815614993, 0.0,
                            0.9541678878162702,
                            5.773159728050814e-15, 1.4595119339458051, 0.0,
                            0.18201910908829552,
                            0.36403821817659104],
                        coverage_override=(maxiters = 6,))
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
