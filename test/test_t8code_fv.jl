module TestExamplesT8codeMesh2D

using Test
using Trixi

include("test_trixi.jl")

# I added this temporary test file for constantly testing while developing.
# The tests have to be adapted at the end.
EXAMPLES_DIR = joinpath(examples_dir(), "t8code_2d_fv")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)
mkdir(outdir)

@testset "T8codeMesh2D" begin
#! format: noindent

# @trixi_testset "test save_mesh_file" begin
#     @test_throws Exception begin
#         # Save mesh file support will be added in the future. The following
#         # lines of code are here for satisfying code coverage.

#         # Create dummy mesh.
#         mesh = T8codeMesh((1, 1), polydeg = 1,
#                           mapping = Trixi.coordinates2mapping((-1.0, -1.0), (1.0, 1.0)),
#                           initial_refinement_level = 1)

#         # This call throws an error.
#         Trixi.save_mesh_file(mesh, "dummy")
#     end
# end

# @trixi_testset "test check_for_negative_volumes" begin
#     @test_warn "Discovered negative volumes" begin
#         # Unstructured mesh with six cells which have left-handed node ordering.
#         mesh_file = Trixi.download("https://gist.githubusercontent.com/jmark/bfe0d45f8e369298d6cc637733819013/raw/cecf86edecc736e8b3e06e354c494b2052d41f7a/rectangle_with_negative_volumes.msh",
#                                    joinpath(EXAMPLES_DIR,
#                                             "rectangle_with_negative_volumes.msh"))

#         # This call should throw a warning about negative volumes detected.
#         mesh = T8codeMesh(mesh_file, 2)
#     end
# end

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
    @trixi_testset "second-order FV, extended reconstruction stencil" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            extended_reconstruction_stencil=true,
                            l2=[0.028436326251639936],
                            linf=[0.08696815845435057])
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
                            l2=[0.5899012906928289],
                            linf=[0.8970164922705812])
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
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_hybrid.jl"),
                            order=1,
                            initial_refinement_level=2,
                            cmesh=Trixi.cmesh_new_periodic_hybrid(),
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
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_hybrid.jl"),
                            order=1,
                            initial_refinement_level=1,
                            cmesh=Trixi.cmesh_new_tri(trees_per_dimension=(2, 2)),
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
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_hybrid.jl"),
                            order=1,
                            initial_refinement_level=2,
                            cmesh = Trixi.cmesh_new_periodic_hybrid2(),
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
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_hybrid.jl"),
                            initial_refinement_level=2,
                            cmesh=Trixi.cmesh_new_periodic_hybrid(),
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
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
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
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
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
    # When fixed, also add to mpi test file.
    # @trixi_testset "second-order FV - triangles" begin
    #     @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
    #                         cmesh=Trixi.cmesh_new_tri(periodicity = (false, false)),
    #                         l2=[0.0],
    #                         linf=[0.0])
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
                                0.031971635993647315,
                                0.016631028330554957,
                                0.016630833188111448,
                                0.04813246238398825,
                            ],
                            linf=[
                                0.055105654108624336,
                                0.03647317645079773,
                                0.03647020577993976,
                                0.08112180586875883,
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
    @trixi_testset "second-order FV, extended reconstruction stencil" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                            order=2,
                            extended_reconstruction_stencil=true,
                            l2=[
                                0.058781550112786296,
                                0.02676026477370241,
                                0.02673090935979779,
                                0.08033279155463603,
                            ],
                            linf=[
                                0.09591263836601072,
                                0.05351985245787505,
                                0.05264935415308125,
                                0.14318629241962988,
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
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_blast_wave.jl"),
                            order=1,
                            l2=[
                                0.5733341919395403,
                                0.11399976571202451,
                                0.11399976571202453,
                                1.3548613737038324,
                            ],
                            linf=[
                                1.732836334678142,
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
    @trixi_testset "second-order FV, extended reconstruction stencil" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_blast_wave.jl"),
                            l2=[
                                0.7331398527938754,
                                0.15194349346989244,
                                0.1519434934698924,
                                1.299914264830515,
                            ],
                            linf=[
                                2.2864127304524726,
                                0.3051023693829176,
                                0.30510236938291757,
                                2.6171402581107936,
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

@trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_kelvin_helmholtz_instability.jl"),
                            order=1,
                            l2=[
                                0.25420413805862135,
                                0.22153054262689362,
                                0.11870842058617848,
                                0.03626117501911353,
                            ],
                            linf=[
                                0.5467894727797227,
                                0.4156157752497065,
                                0.26176691230685767,
                                0.0920609123083227,
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
                                0.020439850138732837,
                            ],
                            linf=[
                                0.5069212604421109,
                                0.365579474379667,
                                0.24226411409222004,
                                0.049201093470609525,
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
                                0.0395037796064376,
                            ],
                            linf=[
                                0.6484515918189779,
                                0.3067327488921227,
                                0.34771083375083534,
                                0.10713502930441887,
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

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
