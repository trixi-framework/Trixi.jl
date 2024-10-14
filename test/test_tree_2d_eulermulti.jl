module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "Compressible Euler Multicomponent" begin
    @trixi_testset "Testing entropy2cons and cons2entropy" begin
        using ForwardDiff
        gammas = (1.1546412974182538, 1.1171560258914812, 1.097107661471476,
                  1.0587601652669245, 1.6209889683979308, 1.6732209755396386,
                  1.2954303574165822)
        gas_constants = (5.969461071171914, 3.6660802003290183, 6.639008614675539,
                         8.116604827140456, 6.190706056680031, 1.6795013743693712,
                         2.197737590916966)
        equations = CompressibleEulerMulticomponentEquations2D(gammas = SVector{length(gammas)}(gammas...),
                                                               gas_constants = SVector{length(gas_constants)}(gas_constants...))
        u = [-1.7433292819144075, 0.8844413258376495, 0.6050737175812364,
            0.8261998359817043, 1.0801186290896465, 0.505654488367698,
            0.6364415555805734, 0.851669392285058, 0.31219606420306223,
            1.0930477805612038]
        w = cons2entropy(u, equations)
        # test that the entropy variables match the gradients of the total entropy
        @test w ≈ ForwardDiff.gradient(u -> Trixi.total_entropy(u, equations), u)
        # test that `entropy2cons` is the inverse of `cons2entropy`
        @test entropy2cons(w, equations) ≈ u
    end

    # NOTE: Some of the L2/Linf errors are comparably large. This is due to the fact that some of the
    #       simulations are set up with dimensional states. For example, the reference pressure in SI
    #       units is 101325 Pa, i.e., pressure has values of O(10^5)

    @trixi_testset "elixir_eulermulti_shock_bubble.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble.jl"),
                            l2=[
                                73.78467629094177,
                                0.9174752929795251,
                                57942.83587826468,
                                0.1828847253029943,
                                0.011127037850925347
                            ],
                            linf=[
                                196.81051991521073,
                                7.8456811648529605,
                                158891.88930113698,
                                0.811379581519794,
                                0.08011973559187913
                            ],
                            tspan=(0.0, 0.001))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_eulermulti_shock_bubble_shockcapturing_subcell_positivity.jl" begin
        rm(joinpath("out", "deviations.txt"), force = true)
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_eulermulti_shock_bubble_shockcapturing_subcell_positivity.jl"),
                            l2=[
                                81.52845664909304,
                                2.5455678559421346,
                                63229.190712645846,
                                0.19929478404550321,
                                0.011068604228443425
                            ],
                            linf=[
                                249.21708417382013,
                                40.33299887640794,
                                174205.0118831558,
                                0.6881458768113586,
                                0.11274401158173972
                            ],
                            initial_refinement_level=3,
                            tspan=(0.0, 0.001),
                            save_errors=true)
        lines = readlines(joinpath("out", "deviations.txt"))
        @test lines[1] == "# iter, simu_time, rho1_min, rho2_min"
        # Runs with and without coverage take 1 and 15 time steps.
        @test startswith(lines[end], "1")
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

    @trixi_testset "elixir_eulermulti_shock_bubble_shockcapturing_subcell_minmax.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_eulermulti_shock_bubble_shockcapturing_subcell_minmax.jl"),
                            l2=[
                                73.41054363926742,
                                1.5072038797716156,
                                57405.58964098063,
                                0.17877099207437225,
                                0.010085388785440972
                            ],
                            linf=[
                                213.59140793740318,
                                24.57625853486584,
                                152498.21319871658,
                                0.5911106543157919,
                                0.09936092838440383
                            ],
                            initial_refinement_level=3,
                            tspan=(0.0, 0.001))
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

    @trixi_testset "elixir_eulermulti_ec.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
                            l2=[
                                0.050182236154087095,
                                0.050189894464434635,
                                0.2258715597305131,
                                0.06175171559771687
                            ],
                            linf=[
                                0.3108124923284472,
                                0.3107380389947733,
                                1.054035804988521,
                                0.29347582879608936
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

    @trixi_testset "elixir_eulermulti_es.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
                            l2=[
                                0.0496546258404055,
                                0.04965550099933263,
                                0.22425206549856372,
                                0.004087155041747821,
                                0.008174310083495642,
                                0.016348620166991283,
                                0.032697240333982566
                            ],
                            linf=[
                                0.2488251110766228,
                                0.24832493304479406,
                                0.9310354690058298,
                                0.017452870465607374,
                                0.03490574093121475,
                                0.0698114818624295,
                                0.139622963724859
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

    @trixi_testset "elixir_eulermulti_convergence_ec.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_ec.jl"),
                            l2=[
                                0.00012290225488326508,
                                0.00012290225488321876,
                                0.00018867397906337653,
                                4.8542321753649044e-5,
                                9.708464350729809e-5
                            ],
                            linf=[
                                0.0006722819239133315,
                                0.0006722819239128874,
                                0.0012662292789555885,
                                0.0002843844182700561,
                                0.0005687688365401122
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

    @trixi_testset "elixir_eulermulti_convergence_es.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
                            l2=[
                                2.2661773867001696e-6,
                                2.266177386666318e-6,
                                6.593514692980009e-6,
                                8.836308667348217e-7,
                                1.7672617334696433e-6
                            ],
                            linf=[
                                1.4713170997993075e-5,
                                1.4713170997104896e-5,
                                5.115618808515521e-5,
                                5.3639516094383666e-6,
                                1.0727903218876733e-5
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

    @trixi_testset "elixir_eulermulti_convergence_es.jl with flux_chandrashekar" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
                            l2=[
                                1.8621737639352465e-6,
                                1.862173764098385e-6,
                                5.942585713809631e-6,
                                6.216263279534722e-7,
                                1.2432526559069443e-6
                            ],
                            linf=[
                                1.6235495582606063e-5,
                                1.6235495576388814e-5,
                                5.854523678827661e-5,
                                5.790274858807898e-6,
                                1.1580549717615796e-5
                            ],
                            volume_flux=flux_chandrashekar)
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
