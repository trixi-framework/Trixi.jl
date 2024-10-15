module TestExamplesMPIT8codeMesh2D

using Test
using Trixi

include("test_trixi.jl")

const EXAMPLES_DIR = pkgdir(Trixi, "examples", "t8code_2d_fv")

@testset "T8codeMesh MPI FV" begin
#! format: noindent

# Run basic tests
@testset "Examples 2D" begin
    # Linear scalar advection
    @trixi_testset "elixir_advection_basic.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                                order=1,
                                l2=[0.08551397247817498],
                                linf=[0.12087467695430498])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        @trixi_testset "second-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                                l2=[0.008142380494734171],
                                linf=[0.018687916234976898])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        # The extended reconstruction stencil is currently not mpi parallel.
        # The current version runs through but an error occurs on some rank.
    end

    @trixi_testset "elixir_advection_gauss.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_gauss.jl"),
                                order=1,
                                l2=[0.5598148317954682],
                                linf=[0.6301130236005371])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        @trixi_testset "second-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_gauss.jl"),
                                l2=[0.5899077806567905],
                                linf=[0.8972489222157533])
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

    @trixi_testset "elixir_advection_basic_hybrid.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_basic_hybrid.jl"),
                                order=1,
                                initial_refinement_level=2,
                                cmesh=Trixi.cmesh_new_hybrid(),
                                l2=[0.2253867410593706],
                                linf=[0.34092690256865166])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        @trixi_testset "first-order FV - triangles" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_basic_hybrid.jl"),
                                order=1,
                                initial_refinement_level=1,
                                cmesh=Trixi.cmesh_new_tri(trees_per_dimension = (2, 2)),
                                l2=[0.29924666807083133],
                                linf=[0.4581996753014146])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        @trixi_testset "first-order FV - hybrid2" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_basic_hybrid.jl"),
                                order=1,
                                initial_refinement_level=2,
                                cmesh=Trixi.cmesh_new_periodic_hybrid2(),
                                l2=[0.20740154468889108],
                                linf=[0.4659917007721659])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        @trixi_testset "second-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_basic_hybrid.jl"),
                                initial_refinement_level=2,
                                cmesh=Trixi.cmesh_new_hybrid(),
                                l2=[0.1296561675517274],
                                linf=[0.25952934874433753])
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

    @trixi_testset "elixir_advection_nonperiodic.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_nonperiodic.jl"),
                                order=1,
                                l2=[0.07215018673798403],
                                linf=[0.12087525707243896])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        @trixi_testset "second-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_nonperiodic.jl"),
                                l2=[0.017076631443535124],
                                linf=[0.05613089948002803])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        # TODO: Somehow, this non-periodic run with a triangular mesh is unstable.
        # When fixed, also add here.
    end

    @trixi_testset "elixir_euler_source_terms.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                                order=1,
                                l2=[
                                    0.059376731961878,
                                    0.019737707470047838,
                                    0.019737707470047747,
                                    0.09982550390697936
                                ],
                                linf=[
                                    0.08501451493301548,
                                    0.029105783468157398,
                                    0.029105783468157842,
                                    0.1451756151490775
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
        @trixi_testset "second-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                                order=2,
                                l2=[
                                    0.031971635416962525,
                                    0.016630983283552267,
                                    0.016630873316960327,
                                    0.04813244762272231
                                ],
                                linf=[
                                    0.055105205670854085,
                                    0.03647221946045942,
                                    0.036470504033139894,
                                    0.0811201478913759
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

    @trixi_testset "elixir_euler_blast_wave.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_blast_wave.jl"),
                            order=1,
                            l2=[
                                0.5733341919395405,
                                0.11399976571202448,
                                0.1139997657120245,
                                1.3548613737038315
                            ],
                            linf=[
                                1.7328363346781415,
                                0.27645456051355827,
                                0.27645456051355827,
                                2.6624886901791407
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

    @trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_euler_kelvin_helmholtz_instability.jl"),
                                order=1,
                                l2=[
                                    0.25420413805862135,
                                    0.22153054262689362,
                                    0.11870842058617848,
                                    0.03626117501911353
                                ],
                                linf=[
                                    0.5467894727797227,
                                    0.4156157752497065,
                                    0.26176691230685767,
                                    0.0920609123083227
                                ],
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
        @trixi_testset "second-order FV - quads" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_euler_kelvin_helmholtz_instability.jl"),
                                order=2,
                                cmesh=Trixi.cmesh_new_quad(periodicity = (true, true)),
                                l2=[
                                    0.2307479238046326,
                                    0.19300139957275295,
                                    0.1176326315506721,
                                    0.020439850138732837
                                ],
                                linf=[
                                    0.5069212604421109,
                                    0.365579474379667,
                                    0.24226411409222004,
                                    0.049201093470609525
                                ],
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
        @trixi_testset "second-order FV - triangles" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_euler_kelvin_helmholtz_instability.jl"),
                                order=2,
                                l2=[
                                    0.16181566308374545,
                                    0.10090964843765918,
                                    0.15229553744179888,
                                    0.0395037796064376
                                ],
                                linf=[
                                    0.6484515918189779,
                                    0.3067327488921227,
                                    0.34771083375083534,
                                    0.10713502930441887
                                ],
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
    end
end
end # T8codeMesh MPI

end # module
