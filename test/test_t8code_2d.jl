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

    @trixi_testset "test save_mesh_file" begin
      @test_throws Exception begin
        # Save mesh file support will be added in the future. The following
        # lines of code are here for satisfying code coverage.

        # Create dummy mesh.
        mesh = T8codeMesh((1, 1), polydeg = 1,
                          mapping = Trixi.coordinates2mapping((-1.0, -1.0),  ( 1.0,  1.0)),
                          initial_refinement_level = 1)

        # This call throws an error.
        Trixi.save_mesh_file(mesh, "dummy")
      end
    end

    @trixi_testset "elixir_advection_basic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            # Expected errors are exactly the same as with TreeMesh!
                            l2=[8.311947673061856e-6],
                            linf=[6.627000273229378e-5])
    end

    @trixi_testset "elixir_advection_nonconforming_flag.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_nonconforming_flag.jl"),
                            l2=[3.198940059144588e-5],
                            linf=[0.00030636069494005547])
    end

    @trixi_testset "elixir_advection_unstructured_flag.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_unstructured_flag.jl"),
                            l2=[0.0005379687442422346],
                            linf=[0.007438525029884735])
    end

    @trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_unstructured_flag.jl"),
                            l2=[0.001993165013217687],
                            linf=[0.032891018571625796],
                            coverage_override=(maxiters = 6,))
    end

    @trixi_testset "elixir_advection_amr_solution_independent.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_solution_independent.jl"),
                            # Expected errors are exactly the same as with StructuredMesh!
                            l2=[4.949660644033807e-5],
                            linf=[0.0004867846262313763],
                            coverage_override=(maxiters = 6,))
    end

    @trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_flag.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_nonconforming_unstructured_flag.jl"),
                            l2=[
                                0.0034516244508588046,
                                0.0023420334036925493,
                                0.0024261923964557187,
                                0.004731710454271893,
                            ],
                            linf=[
                                0.04155789011775046,
                                0.024772109862748914,
                                0.03759938693042297,
                                0.08039824959535657,
                            ])
    end

    @trixi_testset "elixir_euler_free_stream.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                            l2=[
                                2.063350241405049e-15,
                                1.8571016296925367e-14,
                                3.1769447886391905e-14,
                                1.4104095258528071e-14,
                            ],
                            linf=[1.9539925233402755e-14, 2e-12, 4.8e-12, 4e-12],
                            atol=2.0e-12,)
    end

    @trixi_testset "elixir_euler_shockcapturing_ec.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
                            l2=[
                                9.53984675e-02,
                                1.05633455e-01,
                                1.05636158e-01,
                                3.50747237e-01,
                            ],
                            linf=[
                                2.94357464e-01,
                                4.07893014e-01,
                                3.97334516e-01,
                                1.08142520e+00,
                            ],
                            tspan=(0.0, 1.0))
    end

    @trixi_testset "elixir_euler_sedov.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                            l2=[
                                3.76149952e-01,
                                2.46970327e-01,
                                2.46970327e-01,
                                1.28889042e+00,
                            ],
                            linf=[
                                1.22139001e+00,
                                1.17742626e+00,
                                1.17742626e+00,
                                6.20638482e+00,
                            ],
                            tspan=(0.0, 0.3))
    end

    @trixi_testset "elixir_shallowwater_source_terms.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                            l2=[
                                9.168126407325352e-5,
                                0.0009795410115453788,
                                0.002546408320320785,
                                3.941189812642317e-6,
                            ],
                            linf=[
                                0.0009903782521019089,
                                0.0059752684687262025,
                                0.010941106525454103,
                                1.2129488214718265e-5,
                            ],
                            tspan=(0.0, 0.1))
    end

    @trixi_testset "elixir_mhd_alfven_wave.jl" begin
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
    end

    @trixi_testset "elixir_mhd_rotor.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
                            l2=[0.44211360369891683, 0.8805178316216257, 0.8262710688468049,
                                0.0,
                                0.9616090460973586, 0.10386643568745411,
                                0.15403457366543802, 0.0,
                                2.8399715649715473e-5],
                            linf=[10.04369305341599, 17.995640564998403, 9.576041548174265,
                                0.0,
                                19.429658884314534, 1.3821395681242314, 1.818559351543182,
                                0.0,
                                0.002261930217575465],
                            tspan=(0.0, 0.02))
    end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
