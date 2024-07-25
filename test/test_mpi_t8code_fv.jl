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
                                order=2,
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
        # The extended reconstruction stencil currently is not mpi parallel.
        # The current version runs through but an error occurs on some rank (somehow not printed to the terminal).
        # @trixi_testset "second-order FV, extended reconstruction stencil" begin
        #     @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
        #                         order=2,
        #                         extended_reconstruction_stencil=true,
        #                         l2=[0.020331012873518642],
        #                         linf=[0.05571209803860677])
        #     # Ensure that we do not have excessive memory allocations
        #     # (e.g., from type instabilities)
        #     let
        #         t = sol.t[end]
        #         u_ode = sol.u[end]
        #         du_ode = similar(u_ode)
        #         @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        #     end
        # end
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

    @trixi_testset "elixir_euler_source_terms.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                                order=1,
                                l2=[
                                    0.059376731961878,
                                    0.019737707470047838,
                                    0.019737707470047747,
                                    0.09982550390697936,
                                ],
                                linf=[
                                    0.08501451493301548,
                                    0.029105783468157398,
                                    0.029105783468157842,
                                    0.1451756151490775,
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
                                    0.04813244762272231,
                                ],
                                linf=[
                                    0.055105205670854085,
                                    0.03647221946045942,
                                    0.036470504033139894,
                                    0.0811201478913759,
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
                                1.3548613737038315,
                            ],
                            linf=[
                                1.7328363346781415,
                                0.27645456051355827,
                                0.27645456051355827,
                                2.6624886901791407,
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
                                    0.2542045564471016,
                                    0.22153069577606582,
                                    0.11870840559952726,
                                    0.03626114330454897,
                                ],
                                linf=[
                                    0.5467901048636064,
                                    0.4156157765819209,
                                    0.26176688262532194,
                                    0.0920608815870434,
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
        @trixi_testset "second-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_euler_kelvin_helmholtz_instability.jl"),
                                order=2,
                                cmesh=Trixi.cmesh_quad(periodicity = (true, true)),
                                l2=[
                                    0.2307463806750956,
                                    0.19300049982364456,
                                    0.11761779688825669,
                                    0.020439515334349196,
                                ],
                                linf=[
                                    0.5069318322100549,
                                    0.365584194964112,
                                    0.24224645130314845,
                                    0.04918806108930651,
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
