module TestExamplesT8codeMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "t8code_3d_fv")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)
mkdir(outdir)

@testset "T8codeMesh3D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            order=1,
                            initial_refinement_level=4,
                            l2=[0.2848617953369851],
                            linf=[0.3721898718954475])
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
                            initial_refinement_level=4,
                            l2=[0.10381089565603231],
                            linf=[0.13787405651527007])
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
                            initial_refinement_level=3,
                            extended_reconstruction_stencil=true,
                            l2=[0.3282177575292713],
                            linf=[0.39002345444858333])
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
                            l2=[0.03422041780251932],
                            linf=[0.7655163504661142])
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
                            l2=[0.02129708543319383],
                            linf=[0.5679262915623222])
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

@trixi_testset "elixir_advection_amr.jl" begin
    @trixi_testset "first-order FV" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr.jl"),
                            order=1,
                            l2=[0.036187250213578236],
                            linf=[0.2352594244477838])
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
                                     "elixir_advection_amr.jl"),
                            l2=[0.013166109132824377],
                            linf=[0.08606560587093615])
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
                                     "elixir_advection_amr.jl"),
                            extended_reconstruction_stencil=true,
                            l2=[0.01826381435027002],
                            linf=[0.12024798441006923])
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
                            l2=[0.20282363730327146],
                            linf=[0.28132446651281295])
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
                            l2=[0.02153993127089835],
                            linf=[0.039109618097251886])
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
                        l2=[0.022202106950138526],
                        linf=[0.0796166790338586])
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
                                0.050763790354290725,
                                0.0351299673616484,
                                0.0351299673616484,
                                0.03512996736164839,
                                0.1601847269543808
                            ],
                            linf=[
                                0.07175521415072939,
                                0.04648499338897771,
                                0.04648499338897816,
                                0.04648499338897816,
                                0.2235470564880404
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
                                0.012308219704695382,
                                0.010791416898840429,
                                0.010791416898840464,
                                0.010791416898840377,
                                0.036995680347196136
                            ],
                            linf=[
                                0.01982294164697862,
                                0.01840725612418126,
                                0.01840725612418148,
                                0.01840725612418148,
                                0.05736595182767079
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
                                0.05057867333486591,
                                0.03596196296013507,
                                0.03616867188152877,
                                0.03616867188152873,
                                0.14939041550302212
                            ],
                            linf=[
                                0.07943789383956079,
                                0.06389365911606859,
                                0.06469291944863809,
                                0.0646929194486372,
                                0.23507781748792533
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
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
