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
                            linf=[0.0004867846262313763])

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
                            linf=[0.01752690016659812])

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
                            linf=[6.21489667023134e-5])

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
                            l2=[
                                0.0322570437144848,
                                0.03598284801272945,
                                0.03562228071357411,
                                0.05288641880143085,
                                0.040752873778199326,
                                0.04207276835260492,
                                0.04171391252403866,
                                0.05289242879893149,
                                0.0016038935411812223
                            ],
                            linf=[
                                0.175984910510666,
                                0.13999726708245439,
                                0.13336032728399658,
                                0.21248359539637798,
                                0.133294808938885,
                                0.17934684696413217,
                                0.1831567822932948,
                                0.21575881133569155,
                                0.01967917976620703
                            ],
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
                            0.02918489280986591,
                            0.03430197485998593,
                            0.033958258414233555,
                            0.05015261058202805,
                            0.03823791246217151,
                            0.04013747495809935,
                            0.039752335789428925,
                            0.05021821589628951,
                            0.002332549196680272
                        ],
                        linf=[
                            0.1031205532032593,
                            0.1408724803980772,
                            0.1377071548888711,
                            0.24265946222482881,
                            0.12403522681540824,
                            0.19380378175033763,
                            0.19743139999820714,
                            0.2453908414271658,
                            0.019597656830548395
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
