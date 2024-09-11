module TestTree2DFDSBP

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_fdsbp")

@testset "Linear scalar advection" begin
#! format: noindent

@trixi_testset "elixir_advection_extended.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[2.898644263922225e-6],
                        linf=[8.491517930142578e-6],
                        rtol=1.0e-7) # These results change a little bit and depend on the CI system

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_extended.jl with periodic operators" begin
    global D = SummationByPartsOperators.periodic_derivative_operator(derivative_order = 1,
                                                                      accuracy_order = 4,
                                                                      xmin = 0.0,
                                                                      xmax = 1.0,
                                                                      N = 40)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[1.1239649404463432e-5],
                        linf=[1.5895264629195438e-5],
                        D_SBP=D,
                        initial_refinement_level=0)

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

@testset "Compressible Euler" begin
    @trixi_testset "elixir_euler_convergence.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
                            l2=[
                                1.7088389997042244e-6,
                                1.7437997855125774e-6,
                                1.7437997855350776e-6,
                                5.457223460127621e-6
                            ],
                            linf=[
                                9.796504903736292e-6,
                                9.614745892783105e-6,
                                9.614745892783105e-6,
                                4.026107182575345e-5
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

    @trixi_testset "elixir_euler_convergence.jl with Lax-Friedrichs splitting" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
                            l2=[
                                2.1149087345799973e-6,
                                1.9391438806845798e-6,
                                1.9391438806759794e-6,
                                5.842833764682604e-6
                            ],
                            linf=[
                                1.3679037540903494e-5,
                                1.1770587849069258e-5,
                                1.1770587848403125e-5,
                                4.68952678644996e-5
                            ],
                            tspan=(0.0, 0.1), flux_splitting=splitting_lax_friedrichs)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_convergence.jl with Drikakis-Tsangaris splitting" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence.jl"),
                            l2=[
                                1.708838999643608e-6,
                                1.7437997854485807e-6,
                                1.7437997854741082e-6,
                                5.457223460116349e-6
                            ],
                            linf=[
                                9.796504911285808e-6,
                                9.614745899888533e-6,
                                9.614745899444443e-6,
                                4.02610718399643e-5
                            ],
                            tspan=(0.0, 0.1), flux_splitting=splitting_drikakis_tsangaris)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_kelvin_helmholtz_instability.jl"),
                            l2=[
                                0.02607850081951497,
                                0.020357717558016252,
                                0.028510191844948945,
                                0.02951535039734857
                            ],
                            linf=[
                                0.12185328623662173,
                                0.1065055387595834,
                                0.06257122956937419,
                                0.11992349951978643
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

    @trixi_testset "elixir_euler_vortex.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
                            l2=[
                                0.0005330228930711585,
                                0.028475888529345014,
                                0.02847513865894387,
                                0.056259951995581196
                            ],
                            linf=[
                                0.007206088611304784,
                                0.31690373882847234,
                                0.31685665067192326,
                                0.7938167296134893
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
