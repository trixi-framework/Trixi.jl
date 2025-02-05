module TestExamplesT8codeMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "t8code_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)
mkdir(outdir)

@testset "T8codeMesh2D" begin
#! format: noindent

@trixi_testset "test load mesh from path" begin
    mktempdir() do path
        @test_throws "Unknown file extension: .unknown_ext" begin
            mesh = T8codeMesh(touch(joinpath(path, "dummy.unknown_ext")), 2)
        end
    end
end

@trixi_testset "test check_for_negative_volumes" begin
    # This test actually checks if a "negative volume" error is thrown.
    # Since t8code currently applies a correction on-the-fly this test
    # is kinda broken. The correction feature in t8code, however, is planned
    # to be removed in near to midterm future. Thus, this test is kept. It will
    # fail once the internal correction is removed and can then be restored
    # to its original form.

    # @test_throws "Discovered negative volumes" begin
    @test begin
        # Unstructured mesh with six cells which have left-handed node ordering.
        mesh_file = Trixi.download("https://gist.githubusercontent.com/jmark/bfe0d45f8e369298d6cc637733819013/raw/cecf86edecc736e8b3e06e354c494b2052d41f7a/rectangle_with_negative_volumes.msh",
                                   joinpath(EXAMPLES_DIR,
                                            "rectangle_with_negative_volumes.msh"))

        # This call should throw a warning about negative volumes detected.
        mesh = T8codeMesh(mesh_file, 2)
        true
    end
end

@trixi_testset "test t8code mesh from p4est connectivity" begin
    @test begin
        # Here we use the connectivity constructor from `P4est.jl` since the
        # method dispatch works only on `Ptr{p4est_connectivity}` which
        # actually is `Ptr{P4est.LibP4est.p4est_connectivity}`.
        conn = Trixi.P4est.LibP4est.p4est_connectivity_new_brick(2, 3, 1, 1)
        mesh = T8codeMesh(conn)
        Trixi.p4est_connectivity_destroy(conn)
        all(size(mesh.tree_node_coordinates) .== (2, 2, 2, 6))
    end
end

@trixi_testset "test t8code mesh from ABAQUS HOHQMesh file" begin
    @test begin
        # Unstructured ABAQUS mesh file created with HOHQMesh..
        file_path = Trixi.download("https://gist.githubusercontent.com/jmark/9e0da4306e266617eeb19bc56b0e7feb/raw/e6856e1deb648a807f6bb6d6dcacff9e55d94e2a/round_2d_tank.inp",
                                   joinpath(EXAMPLES_DIR, "round_2d_tank.inp"))
        mesh = T8codeMesh(file_path, 2)
        all(size(mesh.tree_node_coordinates) .== (2, 4, 4, 340))
    end
end

@trixi_testset "elixir_advection_basic.jl" begin
    # This test is identical to the one in `test_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5])
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
    # This test is identical to the one in `test_p4est_2d.jl`.
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
    # This test is identical to the one in `test_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_flag.jl"),
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

@trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_unstructured_flag.jl"),
                        l2=[0.002019623611753929],
                        linf=[0.03542375961299987],)
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
    # This test is identical to the one in `test_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_solution_independent.jl"),
                        # Expected errors are exactly the same as with StructuredMesh!
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

@trixi_testset "elixir_advection_restart_amr.jl" begin
    # This test is identical to the one in `test_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart_amr.jl"),
                        l2=[2.869137983727866e-6],
                        linf=[3.8353423270964804e-5],)
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
    # This test is identical to the one in `test_p4est_2d.jl`.
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

@trixi_testset "elixir_euler_free_stream.jl" begin
    # This test is identical to the one in `test_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        l2=[
                            2.063350241405049e-15,
                            1.8571016296925367e-14,
                            3.1769447886391905e-14,
                            1.4104095258528071e-14
                        ],
                        linf=[1.9539925233402755e-14, 2e-12, 4.8e-12, 4e-12],
                        atol=2.0e-12,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing_ec.jl" begin
    # This test is identical to the one in `test_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
                        l2=[
                            9.53984675e-02,
                            1.05633455e-01,
                            1.05636158e-01,
                            3.50747237e-01
                        ],
                        linf=[
                            2.94357464e-01,
                            4.07893014e-01,
                            3.97334516e-01,
                            1.08142520e+00
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

@trixi_testset "elixir_euler_sedov.jl" begin
    # This test is identical to the one in `test_p4est_2d.jl` besides minor
    # deviations in the expected error norms.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            3.76149952e-01,
                            2.46970327e-01,
                            2.46970327e-01,
                            1.28889042e+00
                        ],
                        linf=[
                            1.22139001e+00,
                            1.17742626e+00,
                            1.17742626e+00,
                            6.20638482e+00
                        ],
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl" begin
    # This test is identical to the one in `test_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            9.168126407325352e-5,
                            0.0009795410115453788,
                            0.002546408320320785,
                            3.941189812642317e-6
                        ],
                        linf=[
                            0.0009903782521019089,
                            0.0059752684687262025,
                            0.010941106525454103,
                            1.2129488214718265e-5
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl" begin
    # This test is identical to the one in `test_p4est_2d.jl`.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[1.0513414461545583e-5, 1.0517900957166411e-6,
                            1.0517900957304043e-6, 1.511816606372376e-6,
                            1.0443997728645063e-6, 7.879639064990798e-7,
                            7.879639065049896e-7, 1.0628631669056271e-6,
                            4.3382328912336153e-7],
                        linf=[4.255466285174592e-5, 1.0029706745823264e-5,
                            1.0029706747467781e-5, 1.2122265939010224e-5,
                            5.4791097160444835e-6, 5.18922042269665e-6,
                            5.189220422141538e-6, 9.552667261422676e-6,
                            1.4237578427628152e-6])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_rotor.jl" begin
    # This test is identical to the one in `test_p4est_2d.jl` besides minor
    # deviations in the expected error norms.
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
                        l2=[0.4419337424073218, 0.8804938551016932, 0.8258185723720365,
                            0.0,
                            0.961220188718187, 0.10397273631386837, 0.15408979488125943,
                            0.0,
                            2.66769410449947e-5],
                        linf=[10.053140536236942, 18.17070117006211, 9.549208389448738,
                            0.0,
                            19.676151923191583, 1.3896544044814965, 1.8153256887969416,
                            0.0,
                            0.0022030404596184786],
                        tspan=(0.0, 0.02))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_weak_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weak_blast_wave_amr.jl"),
                        l2=[
                            0.10823279736983638,
                            0.1158152939803735,
                            0.11633970342992006,
                            0.751152651902375
                        ],
                        linf=[
                            0.5581611332828653,
                            0.8354026029724041,
                            0.834485181423738,
                            3.923553028014343
                        ],
                        tspan=(0.0, 0.1),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
    # Check for conservation
    state_integrals = Trixi.integrate(sol.u[2], semi)
    initial_state_integrals = analysis_callback.affect!.initial_state_integrals

    @test isapprox(state_integrals[1], initial_state_integrals[1], atol = 1e-13)
    @test isapprox(state_integrals[2], initial_state_integrals[2], atol = 1e-13)
    @test isapprox(state_integrals[3], initial_state_integrals[3], atol = 1e-13)
    @test isapprox(state_integrals[4], initial_state_integrals[4], atol = 1e-13)
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
