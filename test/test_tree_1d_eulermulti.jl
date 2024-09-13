module TestExamples1DEulerMulti

using Test
using Trixi
using ForwardDiff

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Compressible Euler Multicomponent" begin
    @trixi_testset "Testing entropy2cons and cons2entropy" begin
        using ForwardDiff
        gammas = (1.3272378792562836, 1.5269959187969864, 1.8362285750521512,
                  1.0409061360276926, 1.4652015053812224, 1.3626493264184423)
        gas_constants = (1.817636851910076, 6.760820475922636, 5.588953939749113,
                         6.31574782981543, 3.362932038038397, 3.212779569399733)
        equations = CompressibleEulerMulticomponentEquations1D(gammas = SVector{length(gammas)}(gammas...),
                                                               gas_constants = SVector{length(gas_constants)}(gas_constants...))
        u = [-1.4632513788889214, 0.9908786980927811, 0.2909066990257628,
            0.6256623915420473, 0.4905882754313441, 0.14481800501749112,
            1.0333532872771651, 0.6805599818745411]
        w = cons2entropy(u, equations)
        # test that the entropy variables match the gradients of the total entropy
        @test w ≈ ForwardDiff.gradient(u -> Trixi.total_entropy(u, equations), u)
        # test that `entropy2cons` is the inverse of `cons2entropy`
        @test entropy2cons(w, equations) ≈ u
    end

    @trixi_testset "elixir_eulermulti_ec.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
                            l2=[0.15330089521538684, 0.4417674632047301,
                                0.016888510510282385, 0.03377702102056477,
                                0.06755404204112954],
                            linf=[0.29130548795961864, 0.8847009003152357,
                                0.034686525099975274, 0.06937305019995055,
                                0.1387461003999011])
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
                                0.1522380497572071,
                                0.43830846465313206,
                                0.03907262116499431,
                                0.07814524232998862
                            ],
                            linf=[
                                0.24939193075537294,
                                0.7139395740052739,
                                0.06324208768391237,
                                0.12648417536782475
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
                                8.575236038539227e-5,
                                0.00016387804318585358,
                                1.9412699303977585e-5,
                                3.882539860795517e-5
                            ],
                            linf=[
                                0.00030593277277124464,
                                0.0006244803933350696,
                                7.253121435135679e-5,
                                0.00014506242870271358
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
                            l2=[1.8983933794407234e-5, 6.207744299844731e-5,
                                1.5466205761868047e-6, 3.0932411523736094e-6,
                                6.186482304747219e-6, 1.2372964609494437e-5],
                            linf=[0.00012014372605895218, 0.0003313207215800418,
                                6.50836791016296e-6, 1.301673582032592e-5,
                                2.603347164065184e-5, 5.206694328130368e-5])
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
                            l2=[1.888450477353845e-5, 5.4910600482795386e-5,
                                9.426737161533622e-7, 1.8853474323067245e-6,
                                3.770694864613449e-6, 7.541389729226898e-6],
                            linf=[0.00011622351152063004, 0.0003079221967086099,
                                3.2177423254231563e-6, 6.435484650846313e-6,
                                1.2870969301692625e-5, 2.574193860338525e-5],
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

    @trixi_testset "elixir_eulermulti_two_interacting_blast_waves.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_eulermulti_two_interacting_blast_waves.jl"),
                            l2=[1.288867611915533, 82.71335258388848, 0.00350680272313187,
                                0.013698784353152794,
                                0.019179518517518084],
                            linf=[29.6413044707026, 1322.5844802186496, 0.09191919374782143,
                                0.31092970966717925,
                                0.4417989757182038],
                            tspan=(0.0, 0.0001))
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
