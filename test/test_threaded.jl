module TestExamplesThreaded

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

@testset "Threaded tests" begin
#! format: noindent

@testset "TreeMesh" begin
    @trixi_testset "elixir_advection_restart.jl" begin
        elixir = joinpath(examples_dir(), "tree_2d_dgsem",
                          "elixir_advection_extended.jl")
        Trixi.mpi_isroot() && println("═"^100)
        Trixi.mpi_isroot() && println(elixir)
        trixi_include(@__MODULE__, elixir, tspan = (0.0, 10.0))
        l2_expected, linf_expected = analysis_callback(sol)

        elixir = joinpath(examples_dir(), "tree_2d_dgsem",
                          "elixir_advection_restart.jl")
        Trixi.mpi_isroot() && println("═"^100)
        Trixi.mpi_isroot() && println(elixir)
        # Errors are exactly the same as in the elixir_advection_extended.jl
        trixi_include(@__MODULE__, elixir)
        l2_actual, linf_actual = analysis_callback(sol)

        Trixi.mpi_isroot() && @test l2_actual == l2_expected
        Trixi.mpi_isroot() && @test linf_actual == linf_expected

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_advection_restart.jl with threaded time integration" begin
        @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                     "elixir_advection_restart.jl"),
                            alg=CarpenterKennedy2N54(williamson_condition = false,
                                                     thread = OrdinaryDiffEq.True()),
                            # Expected errors are exactly the same as in the serial test!
                            l2=[8.005068880114254e-6],
                            linf=[6.39093577996519e-5])
    end

    @trixi_testset "elixir_advection_amr_refine_twice.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                     "elixir_advection_amr_refine_twice.jl"),
                            l2=[0.00020547512522578292],
                            linf=[0.007831753383083506])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_advection_amr_coarsen_twice.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                     "elixir_advection_amr_coarsen_twice.jl"),
                            l2=[0.0014321062757891826],
                            linf=[0.0253454486893413])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                     "elixir_euler_source_terms_nonperiodic.jl"),
                            l2=[
                                2.259440511766445e-6,
                                2.318888155713922e-6,
                                2.3188881557894307e-6,
                                6.3327863238858925e-6
                            ],
                            linf=[
                                1.498738264560373e-5,
                                1.9182011928187137e-5,
                                1.918201192685487e-5,
                                6.0526717141407005e-5
                            ],
                            rtol=0.001)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_euler_ec.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                     "elixir_euler_ec.jl"),
                            l2=[
                                0.061751715597716854,
                                0.05018223615408711,
                                0.05018989446443463,
                                0.225871559730513
                            ],
                            linf=[
                                0.29347582879608825,
                                0.31081249232844693,
                                0.3107380389947736,
                                1.0540358049885143
                            ])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_advection_diffusion.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "tree_2d_dgsem",
                                     "elixir_advection_diffusion.jl"),
                            initial_refinement_level=2, tspan=(0.0, 0.4), polydeg=5,
                            alg=RDPK3SpFSAL49(thread = OrdinaryDiffEq.True()),
                            l2=[4.0915532997994255e-6],
                            linf=[2.3040850347877395e-5])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "FDSBP, elixir_advection_extended.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "tree_2d_fdsbp",
                                     "elixir_advection_extended.jl"),
                            l2=[2.898644263922225e-6],
                            linf=[8.491517930142578e-6],
                            rtol=1.0e-7) # These results change a little bit and depend on the CI system

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "FDSBP, elixir_euler_convergence.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "tree_2d_fdsbp",
                                     "elixir_euler_convergence.jl"),
                            l2=[
                                1.7088389997042244e-6,
                                1.7437997855125774e-6,
                                1.7437997855350776e-6,
                                5.457223460127621e-6
                            ],
                            linf=[
                                9.796504903736292e-6,
                                9.614745892783105e-6,
                                9.614745892783105e-6,
                                4.026107182575345e-5
                            ],
                            tspan=(0.0, 0.1))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end
end

@testset "StructuredMesh" begin
    @trixi_testset "elixir_advection_restart.jl with waving flag mesh" begin
        @test_trixi_include(joinpath(examples_dir(), "structured_2d_dgsem",
                                     "elixir_advection_restart.jl"),
                            l2=[0.00016265538265929818],
                            linf=[0.0015194252169410394],
                            rtol=5.0e-5, # Higher tolerance to make tests pass in CI (in particular with macOS)
                            elixir_file="elixir_advection_waving_flag.jl",
                            restart_file="restart_000000021.h5",
                            # With the default `maxiters = 1` in coverage tests,
                            # there would be no time steps after the restart.
                            coverage_override=(maxiters = 100_000,))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_mhd_ec.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "structured_2d_dgsem",
                                     "elixir_mhd_ec.jl"),
                            l2=[0.04937478399958968, 0.0611701500558669,
                                0.06099805934392425, 0.031551737882277144,
                                0.23191853685798858, 0.02476297013104899,
                                0.024482975007695532, 0.035440179203707095,
                                0.0016002328034991635],
                            linf=[0.24744671083295033, 0.2990591185187605,
                                0.3968520446251412, 0.2226544553988576,
                                0.9752669317263143, 0.12117894533967843,
                                0.12845218263379432, 0.17795590713819576,
                                0.0348517136607105],
                            tspan=(0.0, 0.3))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end
end

@testset "UnstructuredMesh" begin
    @trixi_testset "elixir_acoustics_gauss_wall.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_dgsem",
                                     "elixir_acoustics_gauss_wall.jl"),
                            l2=[0.029330394861252995, 0.029345079728907965,
                                0.03803795043486467, 0.0,
                                7.175152371650832e-16, 1.4350304743301665e-15,
                                1.4350304743301665e-15],
                            linf=[0.36236334472179443, 0.3690785638275256,
                                0.8475748723784078, 0.0,
                                8.881784197001252e-16, 1.7763568394002505e-15,
                                1.7763568394002505e-15],
                            tspan=(0.0, 5.0))

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end
end

@testset "P4estMesh" begin
    @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_flag.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
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
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_eulergravity_convergence.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "p4est_2d_dgsem",
                                     "elixir_eulergravity_convergence.jl"),
                            l2=[
                                0.00024871265138964204,
                                0.0003370077102132591,
                                0.0003370077102131964,
                                0.0007231525513793697
                            ],
                            linf=[
                                0.0015813032944647087,
                                0.0020494288423820173,
                                0.0020494288423824614,
                                0.004793821195083758
                            ],
                            tspan=(0.0, 0.1))
    end
end

@testset "T8codeMesh" begin
    @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_flag.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "t8code_2d_dgsem",
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
    end

    @trixi_testset "elixir_eulergravity_convergence.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "t8code_2d_dgsem",
                                     "elixir_eulergravity_convergence.jl"),
                            l2=[
                                0.00024871265138964204,
                                0.0003370077102132591,
                                0.0003370077102131964,
                                0.0007231525513793697
                            ],
                            linf=[
                                0.0015813032944647087,
                                0.0020494288423820173,
                                0.0020494288423824614,
                                0.004793821195083758
                            ],
                            tspan=(0.0, 0.1))
    end
end

@testset "DGMulti" begin
    @trixi_testset "elixir_euler_weakform.jl (SBP, EC)" begin
        @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                     "elixir_euler_weakform.jl"),
                            cells_per_dimension=(4, 4),
                            volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha),
                            surface_integral=SurfaceIntegralWeakForm(flux_ranocha),
                            approximation_type=SBP(),
                            l2=[
                                0.006400337855843578,
                                0.005303799804137764,
                                0.005303799804119745,
                                0.013204169007030144
                            ],
                            linf=[
                                0.03798302318566282,
                                0.05321027922532284,
                                0.05321027922605448,
                                0.13392025411839015
                            ],)

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_euler_curved.jl with threaded time integration" begin
        @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                     "elixir_euler_curved.jl"),
                            alg=RDPK3SpFSAL49(thread = OrdinaryDiffEq.True()),
                            l2=[
                                1.7204593127904542e-5,
                                1.5921547179522804e-5,
                                1.5921547180107928e-5,
                                4.894071422525737e-5
                            ],
                            linf=[
                                0.00010525416930584619,
                                0.00010003778091061122,
                                0.00010003778085621029,
                                0.00036426282101720275
                            ])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_euler_triangulate_pkg_mesh.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                     "elixir_euler_triangulate_pkg_mesh.jl"),
                            l2=[
                                2.344076909832665e-6,
                                1.8610002398709756e-6,
                                2.4095132179484066e-6,
                                6.37330249340445e-6
                            ],
                            linf=[
                                2.509979394305084e-5,
                                2.2683711321080935e-5,
                                2.6180377720841363e-5,
                                5.575278031910713e-5
                            ])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end

    @trixi_testset "elixir_euler_fdsbp_periodic.jl" begin
        @test_trixi_include(joinpath(examples_dir(), "dgmulti_2d",
                                     "elixir_euler_fdsbp_periodic.jl"),
                            l2=[
                                1.3333320340010056e-6,
                                2.044834627970641e-6,
                                2.044834627855601e-6,
                                5.282189803559564e-6
                            ],
                            linf=[
                                2.7000151718858945e-6,
                                3.988595028259212e-6,
                                3.9885950273710336e-6,
                                8.848583042286862e-6
                            ])

        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 5000
        end
    end
end
end

# Clean up afterwards: delete Trixi.jl output directory
Trixi.mpi_isroot() && isdir(outdir) && @test_nowarn rm(outdir, recursive = true)
Trixi.MPI.Barrier(Trixi.mpi_comm())

end # module
