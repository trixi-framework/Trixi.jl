module TestExamples2DShallowWater

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Shallow Water" begin
#! format: noindent

@trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
                        l2=[
                            0.9911802019934329,
                            0.7340106828033273,
                            0.7446338002084801,
                            0.5875351036989047
                        ],
                        linf=[
                            2.0120253138457564,
                            2.991158989293406,
                            2.6557412817714035,
                            3.0
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            0.9130579602987144,
                            1.0602847041965408e-14,
                            1.082225645390032e-14,
                            0.9130579602987147
                        ],
                        linf=[
                            2.113062037615659,
                            4.6613606802974e-14,
                            5.4225772771633196e-14,
                            2.1130620376156584
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_well_balanced_wall.jl"),
                        l2=[
                            0.9130579602987144,
                            1.0602847041965408e-14,
                            1.082225645390032e-14,
                            0.9130579602987147
                        ],
                        linf=[
                            2.113062037615659,
                            4.6613606802974e-14,
                            5.4225772771633196e-14,
                            2.1130620376156584
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            0.9130579602987147,
                            9.68729463970494e-15,
                            9.694538537436981e-15,
                            0.9130579602987147
                        ],
                        linf=[
                            2.1130620376156584,
                            2.3875905654916432e-14,
                            2.2492839032269154e-14,
                            2.1130620376156584
                        ],
                        surface_flux=(FluxHydrostaticReconstruction(flux_lax_friedrichs,
                                                                    hydrostatic_reconstruction_audusse_etal),
                                      flux_nonconservative_audusse_etal),
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced.jl with flux_nonconservative_wintermeyer_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            0.9130579602987146,
                            1.0323158914614244e-14,
                            1.0276096319430528e-14,
                            0.9130579602987147
                        ],
                        linf=[
                            2.11306203761566,
                            4.063916419044386e-14,
                            3.694484044448245e-14,
                            2.1130620376156584
                        ],
                        surface_flux=(flux_wintermeyer_etal,
                                      flux_nonconservative_wintermeyer_etal),
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.001868474306068482,
                            0.01731687445878443,
                            0.017649083171490863,
                            6.274146767717023e-5
                        ],
                        linf=[
                            0.016962486402209986,
                            0.08768628853889782,
                            0.09038488750767648,
                            0.0001819675955490041
                        ],
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms_dirichlet.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_source_terms_dirichlet.jl"),
                        l2=[
                            0.0018596727473552813,
                            0.017306217777629147,
                            0.016367646997420396,
                            6.274146767723934e-5
                        ],
                        linf=[
                            0.016548007102923368,
                            0.08726160568822783,
                            0.09043621622245013,
                            0.0001819675955490041
                        ],
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.0018952610547425214,
                            0.016943425162728183,
                            0.017556784292859465,
                            6.274146767717414e-5
                        ],
                        linf=[
                            0.0151635341334182,
                            0.07967467926956129,
                            0.08400050790965174,
                            0.0001819675955490041
                        ],
                        tspan=(0.0, 0.025),
                        surface_flux=(flux_hll,
                                      flux_nonconservative_fjordholm_etal))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl with FluxHLL(min_max_speed_naive)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.0018957692481057034,
                            0.016943229710439864,
                            0.01755623297390675,
                            6.274146767717414e-5
                        ],
                        linf=[
                            0.015156105797771602,
                            0.07964811135780492,
                            0.0839787097210376,
                            0.0001819675955490041
                        ],
                        tspan=(0.0, 0.025),
                        surface_flux=(FluxHLL(min_max_speed_naive),
                                      flux_nonconservative_fjordholm_etal))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl with flux_nonconservative_wintermeyer_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.002471853426064005,
                            0.05619168608950033,
                            0.11844727575152562,
                            6.274146767730281e-5
                        ],
                        linf=[
                            0.014332922987500218,
                            0.2141204806174546,
                            0.5392313755637872,
                            0.0001819675955490041
                        ],
                        surface_flux=(flux_wintermeyer_etal,
                                      flux_nonconservative_wintermeyer_etal),
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_wall.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_wall.jl"),
                        l2=[
                            0.1351723240085936,
                            0.20010881416550014,
                            0.2001088141654999,
                            2.719538414346464e-7
                        ],
                        linf=[
                            0.5303608302490757,
                            0.5080987791967457,
                            0.5080987791967506,
                            1.1301675764130437e-6
                        ],
                        tspan=(0.0, 0.25))
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
