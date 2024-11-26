module TestExamplesMPIT8codeMesh2D

using Test
using Trixi

include("test_trixi.jl")

const EXAMPLES_DIR = pkgdir(Trixi, "examples", "t8code_3d_fv")

@testset "T8codeMesh MPI FV" begin
#! format: noindent

# Run basic tests
@testset "Examples 3D" begin
    @trixi_testset "elixir_advection_basic.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                                order=1,
                                initial_refinement_level=4,
                                l2=[0.2848617953369851],
                                linf=[0.3721898718954475])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            # let
            #     t = sol.t[end]
            #     u_ode = sol.u[end]
            #     du_ode = similar(u_ode)
            #     @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            # end
        end
        @trixi_testset "second-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                                initial_refinement_level=4,
                                l2=[0.10381089565603231],
                                linf=[0.13787405651527007])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            # let
            #     t = sol.t[end]
            #     u_ode = sol.u[end]
            #     du_ode = similar(u_ode)
            #     @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            # end
        end
        # The extended reconstruction stencil is currently not mpi parallel.
        # The current version runs through but an error occurs on some rank.
    end

    @trixi_testset "elixir_advection_basic_hybrid.jl" begin
        @trixi_testset "first-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_basic_hybrid.jl"),
                                order=1,
                                l2=[0.20282363730327146],
                                linf=[0.28132446651281295])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            # let
            #     t = sol.t[end]
            #     u_ode = sol.u[end]
            #     du_ode = similar(u_ode)
            #     @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            # end
        end
        @trixi_testset "second-order FV" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_advection_basic_hybrid.jl"),
                                l2=[0.02153993127089835],
                                linf=[0.039109618097251886])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            # let
            #     t = sol.t[end]
            #     u_ode = sol.u[end]
            #     du_ode = similar(u_ode)
            #     @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            # end
        end
    end

    @trixi_testset "elixir_advection_nonperiodic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
                            l2=[0.022202106950138526],
                            linf=[0.0796166790338586])
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        # let
        #     t = sol.t[end]
        #     u_ode = sol.u[end]
        #     du_ode = similar(u_ode)
        #     @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        # end
    end
end
end # T8codeMesh MPI

end # module
