module TestExamples2DShallowWaterTwoLayer

# TODO: TrixiShallowWater: move two layer tests to new package

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")

@testset "Two-Layer Shallow Water" begin
    @trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_shallowwater_twolayer_convergence.jl"),
                            l2=[0.0004016779699408397, 0.005466339651545468,
                                0.006148841330156112,
                                0.0002882339012602492, 0.0030120142442780313,
                                0.002680752838455618,
                                8.873630921431545e-6],
                            linf=[0.002788654460984752, 0.01484602033450666,
                                  0.017572229756493973,
                                  0.0016010835493927011, 0.009369847995372549,
                                  0.008407961775489636,
                                  3.361991620143279e-5],
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
                            l2=[3.2935164267930016e-16, 4.6800825611195103e-17,
                                4.843057532147818e-17,
                                0.0030769233188015013, 1.4809161150389857e-16,
                                1.509071695038043e-16,
                                0.0030769233188014935],
                            linf=[2.248201624865942e-15, 2.346382070278936e-16,
                                  2.208565017494899e-16,
                                  0.026474051138910493, 9.237568031609006e-16,
                                  7.520758026187046e-16,
                                  0.026474051138910267],
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

    @trixi_testset "elixir_shallowwater_twolayer_well_balanced with flux_lax_friedrichs.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_shallowwater_twolayer_well_balanced.jl"),
                            l2=[2.0525741072929735e-16, 6.000589392730905e-17,
                                6.102759428478984e-17,
                                0.0030769233188014905, 1.8421386173122792e-16,
                                1.8473184927121752e-16,
                                0.0030769233188014935],
                            linf=[7.355227538141662e-16, 2.960836949170518e-16,
                                  4.2726562436938764e-16,
                                  0.02647405113891016, 1.038795478061861e-15,
                                  1.0401789378532516e-15,
                                  0.026474051138910267],
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
