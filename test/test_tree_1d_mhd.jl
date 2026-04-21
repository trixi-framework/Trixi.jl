module TestExamples1DMHD

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "MHD" begin
#! format: noindent

@trixi_testset "elixir_mhd_alfven_wave.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            1.440611823425164e-15,
                            1.1373567770134494e-14,
                            3.024482376149653e-15,
                            2.0553143516814395e-15,
                            3.9938347410210535e-14,
                            3.984545392098788e-16,
                            2.4782402104201577e-15,
                            1.551737464879987e-15
                        ],
                        linf=[
                            1.9984014443252818e-15,
                            1.3405943022348765e-14,
                            3.3584246494910985e-15,
                            3.164135620181696e-15,
                            7.815970093361102e-14,
                            8.881784197001252e-16,
                            2.886579864025407e-15,
                            2.942091015256665e-15
                        ],
                        initial_condition=initial_condition_constant,
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            1.0375628983659061e-5,
                            6.571144191446236e-7,
                            3.5833569836289104e-5,
                            3.583356983615859e-5,
                            5.084863194951084e-6,
                            1.1963224165731992e-16,
                            3.598916927583752e-5,
                            3.598916927594727e-5
                        ],
                        linf=[
                            2.614095879338585e-5,
                            9.577266731216823e-7,
                            0.00012406198007461344,
                            0.00012406198007509917,
                            1.5066209528846741e-5,
                            2.220446049250313e-16,
                            0.00012658678753942054,
                            0.00012658678753908748
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

@trixi_testset "elixir_mhd_alfven_wave.jl with flux_derigs_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            1.4396053943470756e-5,
                            1.1211016739165248e-5,
                            3.577870687983967e-5,
                            3.577870687982181e-5,
                            1.967962220860377e-6,
                            1.1963224165731992e-16,
                            3.583562899483433e-5,
                            3.583562899486565e-5
                        ],
                        linf=[
                            5.830577969345718e-5,
                            3.280495696370357e-5,
                            0.00012279619948236953,
                            0.00012279619948227238,
                            6.978806516122482e-6,
                            2.220446049250313e-16,
                            0.00012564003648959932,
                            0.00012564003648994626
                        ],
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

@trixi_testset "elixir_mhd_alfven_wave.jl with flux_hllc" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            1.036850596986597e-5, 1.965192583650368e-6,
                            3.5882124656715505e-5, 3.5882124656638764e-5,
                            5.270975504780837e-6, 1.1963224165731992e-16,
                            3.595811808912869e-5, 3.5958118089159453e-5
                        ],
                        linf=[
                            2.887280521446378e-5, 7.310580790352001e-6,
                            0.00012390046377899755, 0.00012390046377787345,
                            1.5102711136583125e-5, 2.220446049250313e-16,
                            0.0001261935452181312, 0.0001261935452182006
                        ],
                        surface_flux=flux_hllc)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
                        l2=[
                            0.05815183849746399,
                            0.08166807325621023,
                            0.054659228513541165,
                            0.054659228513541165,
                            0.15578125987042743,
                            4.130462730494e-17,
                            0.05465258887150046,
                            0.05465258887150046
                        ],
                        linf=[
                            0.12165312668363826,
                            0.1901920742264952,
                            0.10059813883022554,
                            0.10059813883022554,
                            0.44079257431070706,
                            1.1102230246251565e-16,
                            0.10528911365809579,
                            0.10528911365809579
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

@trixi_testset "elixir_mhd_briowu_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_briowu_shock_tube.jl"),
                        l2=[
                            0.17477712356961989,
                            0.19489623595086944,
                            0.3596546157640463,
                            0.0,
                            0.3723215736814466,
                            1.2060075775846403e-15,
                            0.36276754492568164,
                            0.0
                        ],
                        linf=[
                            0.5797109945880677,
                            0.4372991899547103,
                            1.0906536287185835,
                            0.0,
                            1.0526758874956808,
                            5.995204332975845e-15,
                            1.5122922036932964,
                            0.0
                        ],
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

@trixi_testset "elixir_mhd_torrilhon_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_torrilhon_shock_tube.jl"),
                        l2=[
                            0.45700904847931145,
                            0.4792535936512035,
                            0.340651203521865,
                            0.4478034694296928,
                            0.9204708961093411,
                            1.3216517820475193e-16,
                            0.28897419402047725,
                            0.25521206483145126
                        ],
                        linf=[
                            1.2185238171352286,
                            0.8913202384963431,
                            0.8488793580488431,
                            0.973083603686,
                            1.660723397705417,
                            2.220446049250313e-16,
                            0.6874726847741993,
                            0.65536978110274
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

@trixi_testset "elixir_mhd_torrilhon_shock_tube.jl (HLLC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_torrilhon_shock_tube.jl"),
                        surface_flux=flux_hllc,
                        l2=[
                            0.4573799618744708, 0.4792633358230866, 0.34064852506872795,
                            0.4479668434955162, 0.9203891782415092,
                            1.3216517820475193e-16, 0.28887826520860815,
                            0.255281629265771
                        ],
                        linf=[
                            1.2382842201671505, 0.8929169308132259, 0.871298623806198,
                            0.9822415614542821, 1.6726170732132717,
                            2.220446049250313e-16, 0.7016155888023747,
                            0.6556091522071984
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

@trixi_testset "elixir_mhd_ryujones_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ryujones_shock_tube.jl"),
                        l2=[
                            0.23469781891518154,
                            0.3916675299696121,
                            0.08245195301016353,
                            0.1745346945706147,
                            0.9606363432904367,
                            6.608258910237605e-17,
                            0.21542929107153735,
                            0.10705457908737925
                        ],
                        linf=[
                            0.6447951791685409,
                            0.9461857095377463,
                            0.35074627554617605,
                            0.8515177411529542,
                            2.0770652030507053,
                            1.1102230246251565e-16,
                            0.49670855513788204,
                            0.24830199967863564
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_shu_osher_shock_tube.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shu_osher_shock_tube.jl"),
                        l2=[
                            1.01126210e+00,
                            8.27157902e+00,
                            1.30882545e+00,
                            0.00000000e+00,
                            5.21930435e+01,
                            6.56538824e-16,
                            1.01022340e+00,
                            0.00000000e+00
                        ],
                        linf=[
                            2.87172004e+00,
                            2.26438057e+01,
                            4.16672442e+00,
                            0.00000000e+00,
                            1.35152372e+02,
                            3.44169138e-15,
                            2.83556069e+00,
                            0.00000000e+00
                        ],
                        tspan=(0.0, 0.2),
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

@trixi_testset "elixir_mhd_shu_osher_shock_tube.jl with flipped shock direction" begin
    # Include this elixir to make `initial_condition_shu_osher_shock_tube_flipped` available, which is used below
    trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shu_osher_shock_tube.jl"),
                  tspan = (0.0, 0.0))
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shu_osher_shock_tube.jl"),
                        l2=[
                            1.01539817e+00,
                            8.29625810e+00,
                            1.29548008e+00,
                            0.00000000e+00,
                            5.23565514e+01,
                            3.18641825e-16,
                            1.00485291e+00,
                            0.00000000e+00
                        ],
                        linf=[
                            2.92876280e+00,
                            2.28341581e+01,
                            4.11643561e+00,
                            0.00000000e+00,
                            1.36966213e+02,
                            1.55431223e-15,
                            2.80548864e+00,
                            0.00000000e+00
                        ],
                        initial_condition=initial_condition_shu_osher_shock_tube_flipped,
                        boundary_conditions=BoundaryConditionDirichlet(initial_condition_shu_osher_shock_tube_flipped),
                        tspan=(0.0, 0.2),
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
