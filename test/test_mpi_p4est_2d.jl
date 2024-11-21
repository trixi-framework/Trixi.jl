module TestExamplesMPIP4estMesh2D

using Test
using Trixi

include("test_trixi.jl")

const EXAMPLES_DIR = pkgdir(Trixi, "examples", "p4est_2d_dgsem")

@testset "P4estMesh MPI 2D" begin
#! format: noindent

# Run basic tests
@testset "Examples 2D" begin
    # Linear scalar advection
    @trixi_testset "elixir_advection_basic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            # Expected errors are exactly the same as with TreeMesh!
                            l2=[8.311947673061856e-6],
                            linf=[6.627000273229378e-5])

        @testset "error-based step size control" begin
            Trixi.mpi_isroot() && println("-"^100)
            Trixi.mpi_isroot() &&
                println("elixir_advection_basic.jl with error-based step size control")

            sol = solve(ode, RDPK3SpFSAL35(); abstol = 1.0e-4, reltol = 1.0e-4,
                        ode_default_options()..., callback = callbacks)
            summary_callback()
            errors = analysis_callback(sol)
            if Trixi.mpi_isroot()
                @test errors.l2≈[3.3022040342579066e-5] rtol=1.0e-4
                @test errors.linf≈[0.00011787417954578494] rtol=1.0e-4
            end
        end

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_nonconforming_flag.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_nonconforming_flag.jl"),
                            l2=[3.198940059144588e-5],
                            linf=[0.00030636069494005547])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_unstructured_flag.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_unstructured_flag.jl"),
                            l2=[0.0005379687442422346],
                            linf=[0.007438525029884735])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_amr_solution_independent.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_solution_independent.jl"),
                            # Expected errors are exactly the same as with TreeMesh!
                            l2=[4.949660644033807e-5],
                            linf=[0.0004867846262313763],
                            coverage_override=(maxiters = 6,))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_unstructured_flag.jl"),
                            l2=[0.0012808538770535593],
                            linf=[0.01752690016659812],
                            coverage_override=(maxiters = 6,))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_restart.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                            l2=[4.507575525876275e-6],
                            linf=[6.21489667023134e-5],
                            # With the default `maxiters = 1` in coverage tests,
                            # there would be no time steps after the restart.
                            coverage_override=(maxiters = 100_000,))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_flag.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
                            l2=[
                                0.0034516244508588046,
                                0.0023420334036925493,
                                0.0024261923964557187,
                                0.004731710454271893
                            ],
                            linf=[
                                0.04155789011775046,
                                0.024772109862748914,
                                0.03759938693042297,
                                0.08039824959535657
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
end # P4estMesh MPI

end # module
