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
                            initial_refinement_level=2,
                            l2=[0.23262794278028587],
                            linf=[0.3278746599076482])
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
                            initial_refinement_level=2,
                            l2=[0.08542305264254077],
                            linf=[0.13412953018860274])
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
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_gauss.jl"),
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

@trixi_testset "elixir_advection_basic_hybrid.jl" begin
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_hybrid.jl"),
                            order=1,
                            initial_refinement_level=2,
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
    @trixi_testset "second-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic_hybrid.jl"),
                            initial_refinement_level=2,
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
end

@trixi_testset "elixir_euler_blast_wave.jl" begin
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

@trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_kelvin_helmholtz_instability.jl"),
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
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
