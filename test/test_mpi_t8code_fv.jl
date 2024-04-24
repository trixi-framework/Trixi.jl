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
                                l2=[0.1419061449384701],
                                linf=[0.2086802087402776])
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
                                order=2,
                                l2=[0.020331012873518642],
                                linf=[0.05571209803860677])
            # Ensure that we do not have excessive memory allocations
            # (e.g., from type instabilities)
            let
                t = sol.t[end]
                u_ode = sol.u[end]
                du_ode = similar(u_ode)
                @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
            end
        end
        @trixi_testset "second-order FV, reconstruction stencil" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                                order=2,
                                extended_reconstruction_stencil=false,
                                l2=[0.013669404880134087],
                                linf=[0.03455999303991164])
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

    @trixi_testset "elixir_euler_blast_wave.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_blast_wave.jl"),
                            l2=[
                                0.49698968976388164,
                                0.16934401479236502,
                                0.16934401479236502,
                                0.6743947137176176,
                            ],
                            linf=[
                                1.1342505243873413,
                                0.43853745700004154,
                                0.4385374570000415,
                                3.009703218658938,
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
end # T8codeMesh MPI

end # module
