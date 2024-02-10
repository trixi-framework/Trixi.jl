module TestExamples1DShallowWaterTwoLayer

# TODO: TrixiShallowWater: move two layer tests to new package

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Shallow Water Two layer" begin
    @trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_shallowwater_twolayer_convergence.jl"),
                            l2=[0.005012009872109003, 0.002091035326731071,
                                0.005049271397924551,
                                0.0024633066562966574, 0.0004744186597732739],
                            linf=[0.0213772149343594, 0.005385752427290447,
                                  0.02175023787351349,
                                  0.008212004668840978, 0.0008992474511784199],
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
                            l2=[0.1000774903431289, 0.5670692949571057, 0.08764242501014498,
                                0.45412307886094555, 0.013638618139749523],
                            linf=[0.586718937495144, 2.1215606128311584, 0.5185911311186155,
                                  1.820382495072612, 0.5],
                            surface_flux=(flux_lax_friedrichs,
                                          flux_nonconservative_ersing_etal),
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
