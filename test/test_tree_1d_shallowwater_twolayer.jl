module TestExamples1DShallowWaterTwoLayer

# TODO: TrixiShallowWater: move two layer tests to new package

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Shallow Water Two layer" begin
#! format: noindent

@trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_twolayer_convergence.jl"),
                        l2=[0.0050681532925156945, 0.002089013899370176,
                            0.005105544300292713, 0.002526442122643306,
                            0.0004744186597732706],
                        linf=[0.022256679217306008, 0.005421833004652266,
                            0.02233993939574197, 0.008765261497422516,
                            0.0008992474511784199],
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

@trixi_testset "elixir_shallowwater_twolayer_convergence.jl with flux_es_fjordholm_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_twolayer_convergence.jl"),
                        l2=[0.0027681377074701345, 0.0018007543202559165,
                            0.0028036917433720576,
                            0.0013980358596935737, 0.0004744186597732706],
                        linf=[0.005699303919826093, 0.006432952918256296,
                            0.0058507082844360125, 0.002717615543961216,
                            0.0008992474511784199],
                        surface_flux=(flux_es_fjordholm_etal,
                                      flux_nonconservative_fjordholm_etal),
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

@trixi_testset "elixir_shallowwater_twolayer_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_twolayer_well_balanced.jl"),
                        l2=[8.949288784402005e-16, 4.0636427176237915e-17,
                            0.001002881985401548,
                            2.133351105037203e-16, 0.0010028819854016578],
                        linf=[2.6229018956769323e-15, 1.878451903240623e-16,
                            0.005119880996670156,
                            8.003199803957679e-16, 0.005119880996670666],
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

@trixi_testset "elixir_shallowwater_twolayer_dam_break.jl with flux_lax_friedrichs" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_twolayer_dam_break.jl"),
                        l2=[0.10010269243463918, 0.5668733957648654,
                            0.08759617327649398,
                            0.4538443183566172, 0.013638618139749523],
                        linf=[
                            0.5854202777756559,
                            2.1278930820498934,
                            0.5193686074348809,
                            1.8071213168086229,
                            0.5,
                        ],
                        surface_flux=(flux_lax_friedrichs,
                                      flux_nonconservative_fjordholm_etal),
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
