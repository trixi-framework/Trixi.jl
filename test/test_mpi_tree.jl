module TestExamplesMPITreeMesh

using Test
using Trixi

include("test_trixi.jl")

const EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

# Needed to skip certain tests on Windows CI
CI_ON_WINDOWS = (get(ENV, "GITHUB_ACTIONS", false) == "true") && Sys.iswindows()

@testset "TreeMesh MPI" begin
#! format: noindent

# Run basic tests
@testset "Examples 2D" begin
    # Linear scalar advection
    @trixi_testset "elixir_advection_basic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                            # Expected errors are exactly the same as in the serial test!
                            l2=[8.311947673061856e-6],
                            linf=[6.627000273229378e-5])
    end

    @trixi_testset "elixir_advection_restart.jl" begin
        using OrdinaryDiffEq: RDPK3SpFSAL49
        Trixi.mpi_isroot() && println("═"^100)
        Trixi.mpi_isroot() &&
            println(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"))
        trixi_include(@__MODULE__,
                      joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                      alg = RDPK3SpFSAL49(), tspan = (0.0, 10.0))
        l2_expected, linf_expected = analysis_callback(sol)

        Trixi.mpi_isroot() && println("═"^100)
        Trixi.mpi_isroot() &&
            println(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"))
        # Errors are exactly the same as in the elixir_advection_extended.jl
        trixi_include(@__MODULE__,
                      joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                      alg = RDPK3SpFSAL49())
        l2_actual, linf_actual = analysis_callback(sol)

        Trixi.mpi_isroot() && @test l2_actual == l2_expected
        Trixi.mpi_isroot() && @test linf_actual == linf_expected
    end

    @trixi_testset "elixir_advection_mortar.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_mortar.jl"),
                            # Expected errors are exactly the same as in the serial test!
                            l2=[0.0015188466707237375],
                            linf=[0.008446655719187679])
    end

    @trixi_testset "elixir_advection_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                            # Expected errors are exactly the same as in the serial test!
                            l2=[4.913300828257469e-5],
                            linf=[0.00045263895394385967],
                            coverage_override=(maxiters = 6,))
    end

    @trixi_testset "elixir_advection_amr_nonperiodic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_nonperiodic.jl"),
                            # Expected errors are exactly the same as in the serial test!
                            l2=[3.2207388565869075e-5],
                            linf=[0.0007508059772436404],
                            coverage_override=(maxiters = 6,))
    end

    @trixi_testset "elixir_advection_restart_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_restart_amr.jl"),
                            l2=[8.018498574373939e-5],
                            linf=[0.0007307237754662355])
    end

    # Linear scalar advection with AMR
    # These example files are only for testing purposes and have no practical use
    @trixi_testset "elixir_advection_amr_refine_twice.jl" begin
        # Here, we also test that SaveSolutionCallback prints multiple mesh files with AMR
        # Start with a clean environment: remove Trixi.jl output directory if it exists
        outdir = "out"
        Trixi.mpi_isroot() && isdir(outdir) && rm(outdir, recursive = true)
        Trixi.MPI.Barrier(Trixi.mpi_comm())
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_refine_twice.jl"),
                            l2=[0.00020547512522578292],
                            linf=[0.007831753383083506],
                            coverage_override=(maxiters = 6,))
        meshfiles = filter(file -> endswith(file, ".h5") && startswith(file, "mesh"),
                           readdir(outdir))
        @test length(meshfiles) > 1
    end

    @trixi_testset "elixir_advection_amr_coarsen_twice.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_advection_amr_coarsen_twice.jl"),
                            l2=[0.0014321062757891826],
                            linf=[0.0253454486893413],
                            coverage_override=(maxiters = 6,))
    end

    # Hyperbolic diffusion
    if !CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` in `test/test_mpi.jl`
        @trixi_testset "elixir_hypdiff_lax_friedrichs.jl" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_hypdiff_lax_friedrichs.jl"),
                                l2=[
                                    0.00015687751816056159,
                                    0.001025986772217084,
                                    0.0010259867722169909
                                ],
                                linf=[
                                    0.0011986956416591976,
                                    0.006423873516411049,
                                    0.006423873516411049
                                ])
        end
    end

    @trixi_testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_hypdiff_harmonic_nonperiodic.jl"),
                            l2=[
                                8.61813235543625e-8,
                                5.619399844542781e-7,
                                5.6193998447443e-7
                            ],
                            linf=[
                                1.124861862180196e-6,
                                8.622436471039663e-6,
                                8.622436470151484e-6
                            ])
    end

    @trixi_testset "elixir_hypdiff_nonperiodic.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
                            l2=[
                                8.523077653955306e-6,
                                2.8779323653065056e-5,
                                5.4549427691297846e-5
                            ],
                            linf=[
                                5.5227409524905013e-5,
                                0.0001454489597927185,
                                0.00032396328684569653
                            ])
    end

    if !CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` in `test/test_mpi.jl`
        @trixi_testset "elixir_hypdiff_godunov.jl" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_godunov.jl"),
                                l2=[
                                    5.868147556427088e-6,
                                    3.80517927324465e-5,
                                    3.805179273249344e-5
                                ],
                                linf=[
                                    3.701965498725812e-5,
                                    0.0002122422943138247,
                                    0.00021224229431116015
                                ],
                                atol=2.0e-12) #= required for CI on macOS =#
        end
    end

    # Compressible Euler
    # Note: Some tests here have manually increased relative tolerances since reduction via MPI can
    #       slightly change the L2 error norms (different floating point truncation errors)
    if !CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` in `test/test_mpi.jl`
        @trixi_testset "elixir_euler_source_terms.jl" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                                l2=[
                                    9.321181253186009e-7,
                                    1.4181210743438511e-6,
                                    1.4181210743487851e-6,
                                    4.824553091276693e-6
                                ],
                                linf=[
                                    9.577246529612893e-6,
                                    1.1707525976012434e-5,
                                    1.1707525976456523e-5,
                                    4.8869615580926506e-5
                                ],
                                rtol=2000 * sqrt(eps()))
        end
    end

    # This example file is only for testing purposes and has no practical use
    if !CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` in `test/test_mpi.jl`
        @trixi_testset "elixir_euler_source_terms_amr_refine_coarsen.jl" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_euler_source_terms_amr_refine_coarsen.jl"),
                                l2=[
                                    4.8226610349853444e-5,
                                    4.117706709270575e-5,
                                    4.1177067092959676e-5,
                                    0.00012205252427437389
                                ],
                                linf=[
                                    0.0003543874851490436,
                                    0.0002973166773747593,
                                    0.0002973166773760916,
                                    0.001154106793870291
                                ],
                                # Let this test run until the end to cover the time-dependent lines
                                # of the indicator and the MPI-specific AMR code.
                                coverage_override=(maxiters = 10^5,))
        end
    end

    if !CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` in `test/test_mpi.jl`
        @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
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
        end
    end

    if !CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` in `test/test_mpi.jl`
        @trixi_testset "elixir_euler_ec.jl" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
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

            @testset "error-based step size control" begin
                Trixi.mpi_isroot() && println("-"^100)
                Trixi.mpi_isroot() &&
                    println("elixir_euler_ec.jl with error-based step size control")

                sol = solve(ode, RDPK3SpFSAL35(); abstol = 1.0e-4, reltol = 1.0e-4,
                            ode_default_options()..., callback = callbacks)
                summary_callback()
                errors = analysis_callback(sol)
                if Trixi.mpi_isroot()
                    @test errors.l2≈[
                        0.061653630426688116,
                        0.05006930431098764,
                        0.05007694316484242,
                        0.22550689872331683
                    ] rtol=1.0e-4
                    @test errors.linf≈[
                        0.28516937484583693,
                        0.2983633696512788,
                        0.297812036335975,
                        1.027368795517512
                    ] rtol=1.0e-4
                end
            end
        end
    end

    @trixi_testset "elixir_euler_vortex.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex.jl"),
                            l2=[
                                0.00013492249515826863,
                                0.006615696236378061,
                                0.006782108219800376,
                                0.016393831451740604
                            ],
                            linf=[
                                0.0020782600954247776,
                                0.08150078921935999,
                                0.08663621974991986,
                                0.2829930622010579
                            ],
                            rtol=0.001)
    end

    @trixi_testset "elixir_euler_vortex_mortar.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_mortar.jl"),
                            # Expected errors are exactly the same as in the serial test!
                            l2=[
                                0.0017208369388227673,
                                0.09628684992237334,
                                0.09620157717330868,
                                0.1758809552387432
                            ],
                            linf=[
                                0.021869936355319086,
                                0.9956698009442038,
                                1.0002507727219028,
                                2.223249697515648
                            ])
    end

    @trixi_testset "elixir_euler_vortex_amr.jl" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_amr.jl"),
                            # Expected errors are exactly the same as in the serial test!
                            l2=[
                                5.051719943432265e-5,
                                0.0022574259317084747,
                                0.0021755998463189713,
                                0.004346492398617521
                            ],
                            linf=[
                                0.0012880114865917447,
                                0.03857193149447702,
                                0.031090457959835893,
                                0.12125130332971423
                            ],
                            coverage_override=(maxiters = 6,))
    end

    if !CI_ON_WINDOWS # see comment on `CI_ON_WINDOWS` in `test/test_mpi.jl`
        @trixi_testset "elixir_euler_vortex_shockcapturing.jl" begin
            @test_trixi_include(joinpath(EXAMPLES_DIR,
                                         "elixir_euler_vortex_shockcapturing.jl"),
                                l2=[
                                    0.0017158367642679273,
                                    0.09619888722871434,
                                    0.09616432767924141,
                                    0.17553381166255197
                                ],
                                linf=[
                                    0.021853862449723982,
                                    0.9878047229255944,
                                    0.9880191167111795,
                                    2.2154030488035588
                                ],
                                rtol=0.001)
        end
    end
end
end # TreeMesh MPI

end # module
