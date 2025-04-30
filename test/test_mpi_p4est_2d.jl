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

            # Use callbacks without stepsize_callback to test error-based step size control
            callbacks = CallbackSet(summary_callback, analysis_callback, save_solution)
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
                            linf=[0.0004867846262313763],)

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
                            linf=[0.01752690016659812],)

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
                            linf=[6.21489667023134e-5],)

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

    # Test MPI-parallel handling of .inp meshes generated by HOHQMesh
    @trixi_testset "elixir_euler_wall_bc_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_wall_bc_amr.jl"),
                            l2=[
                                0.02026685991647352,
                                0.017467584076280237,
                                0.011378371604813321,
                                0.05138942558296091
                            ],
                            linf=[
                                0.35924402060711524,
                                0.32068389566068806,
                                0.2361141752119986,
                                0.9289840057748628
                            ],
                            tspan=(0.0, 0.15))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # Test MPI-parallel handling of .inp meshes NOT generated by HOHQMesh
    @trixi_testset "elixir_euler_SD7003airfoil.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                     "elixir_navierstokes_SD7003airfoil.jl"),
                            semi=SemidiscretizationHyperbolic(mesh, equations,
                                                              initial_condition, solver;
                                                              boundary_conditions = boundary_conditions_hyp),
                            analysis_callback=AnalysisCallback(semi,
                                                               interval = analysis_interval,
                                                               output_directory = "out",
                                                               save_analysis = true),
                            l2=[
                                9.316117984455285e-5,
                                4.539266936628966e-5,
                                8.381576796590632e-5,
                                0.00023437941500203496
                            ],
                            linf=[
                                0.31274105032407307,
                                0.2793016762668701,
                                0.22256470161743136,
                                0.7906704256076251
                            ],
                            tspan=(0.0, 5e-3))
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_mhd_alfven_wave_nonconforming.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_mhd_alfven_wave_nonconforming.jl"),
                            l2=[0.032257043714485005,
                                0.0698809831015213,
                                0.07024507293378073,
                                0.09318700512682686,
                                0.04075287377819964,
                                0.06598033890138222,
                                0.06584394125943109,
                                0.09317325194007701,
                                0.001603893541181234],
                            linf=[0.17598491051066556,
                                0.13831592490115455,
                                0.14124330399841845,
                                0.17293937185553027,
                                0.1332948089388849,
                                0.16128651157312346,
                                0.15572969249532598,
                                0.1810247231315753,
                                0.01967917976620706],
                            tspan=(0.0, 0.25),)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    # Same test as above but with only one tree in the mesh
    # We use it to test meshes with elements of different size in each partition
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.02918489280986602,
                            0.06894247101538993,
                            0.06934211084749892,
                            0.09143968257530088,
                            0.038237912462171675,
                            0.06509909515945271,
                            0.06502196369480336,
                            0.0915205366320386,
                            0.0023325491966802855
                        ],
                        linf=[
                            0.10312055320325908,
                            0.13916440641975747,
                            0.14191886090656713,
                            0.16048337905766766,
                            0.12403522681540824,
                            0.14689365133406318,
                            0.1568420189383094,
                            0.16311092390521648,
                            0.01959765683054841
                        ],
                        tspan=(0.0, 0.25), trees_per_dimension=(1, 1),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end
end # P4estMesh MPI

end # module
