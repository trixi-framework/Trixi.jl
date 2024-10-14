module TestExamplesStructuredMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "structured_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "StructuredMesh2D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
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

@trixi_testset "elixir_advection_float32.jl" begin
    # Expected errors are taken from elixir_advection_basic.jl
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_float32.jl"),
                        # Expected errors are taken from elixir_advection_basic.jl
                        l2=[Float32(8.311947673061856e-6)],
                        linf=[Float32(6.627000273229378e-5)],
                        RealT=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_coupled.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_coupled.jl"),
                        l2=[
                            7.816742843336293e-6,
                            7.816742843340186e-6,
                            7.816742843025513e-6,
                            7.816742843061526e-6
                        ],
                        linf=[
                            6.314906965276812e-5,
                            6.314906965187994e-5,
                            6.31490696496595e-5,
                            6.314906965032563e-5
                        ],
                        coverage_override=(maxiters = 10^5,))

    @testset "analysis_callback(sol) for AnalysisCallbackCoupled" begin
        errors = analysis_callback(sol)
        @test errors.l2≈[
            7.816742843336293e-6,
            7.816742843340186e-6,
            7.816742843025513e-6,
            7.816742843061526e-6
        ] rtol=1.0e-4
        @test errors.linf≈[
            6.314906965276812e-5,
            6.314906965187994e-5,
            6.31490696496595e-5,
            6.314906965032563e-5
        ] rtol=1.0e-4
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

@trixi_testset "elixir_advection_meshview.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_meshview.jl"),
                        l2=[
                            8.311947673083206e-6,
                            8.311947673068427e-6
                        ],
                        linf=[
                            6.627000273318195e-5,
                            6.62700027264096e-5
                        ],
                        coverage_override=(maxiters = 10^5,))

    @testset "analysis_callback(sol) for AnalysisCallbackCoupled" begin
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

@trixi_testset "elixir_advection_extended.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[4.220397559713772e-6],
                        linf=[3.477948874874848e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_extended.jl with polydeg=4" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[5.32996976442737e-7],
                        linf=[4.1344662966569246e-6],
                        atol=1e-12, # required to make CI tests pass on macOS
                        cells_per_dimension=(16, 23),
                        polydeg=4,
                        cfl=1.4)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@testset "elixir_advection_rotated.jl" begin
    @trixi_testset "elixir_advection_rotated.jl with α = 0.0" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
                            # Expected errors are exactly the same as in elixir_advection_basic!
                            l2=[8.311947673061856e-6],
                            linf=[6.627000273229378e-5],
                            alpha=0.0)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_rotated.jl with α = 0.1" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
                            # Expected errors differ only slightly from elixir_advection_basic!
                            l2=[8.3122750550501e-6],
                            linf=[6.626802581322089e-5],
                            alpha=0.1)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_advection_rotated.jl with α = 0.5 * pi" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
                            # Expected errors are exactly the same as in elixir_advection_basic!
                            l2=[8.311947673061856e-6],
                            linf=[6.627000273229378e-5],
                            alpha=0.5 * pi)
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

@trixi_testset "elixir_advection_parallelogram.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_parallelogram.jl"),
                        # Expected errors are exactly the same as in elixir_advection_basic!
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

@trixi_testset "elixir_advection_waving_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_waving_flag.jl"),
                        l2=[0.00018553859900545866],
                        linf=[0.0016167719118129753])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream.jl"),
                        l2=[6.8925194184204476e-15],
                        linf=[9.903189379656396e-14])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
                        l2=[0.00025552740731641223],
                        linf=[0.007252625722805939])
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
                        l2=[4.219208035582454e-6],
                        linf=[3.438434404412494e-5],
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

@trixi_testset "elixir_advection_restart.jl with waving flag mesh" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_restart.jl with free stream mesh" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[7.841217436552029e-15],
                        linf=[1.0857981180834031e-13],
                        elixir_file="elixir_advection_free_stream.jl",
                        restart_file="restart_000000036.h5",
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

@trixi_testset "elixir_eulermulti_convergence_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_ec.jl"),
                        l2=[
                            1.5123651627525257e-5,
                            1.51236516273878e-5,
                            2.4544918394022538e-5,
                            5.904791661362391e-6,
                            1.1809583322724782e-5
                        ],
                        linf=[
                            8.393471747591974e-5,
                            8.393471748258108e-5,
                            0.00015028562494778797,
                            3.504466610437795e-5,
                            7.00893322087559e-5
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

@trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
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

@testset "elixir_euler_source_terms_rotated.jl" begin
    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.0" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_rotated.jl"),
                            # Expected errors are exactly the same as in elixir_euler_source_terms!
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
                            alpha=0.0)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.1" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_rotated.jl"),
                            # Expected errors differ only slightly from elixir_euler_source_terms!
                            l2=[
                                9.321188057029291e-7,
                                1.3195106906473365e-6,
                                1.510307360354032e-6,
                                4.82455408101712e-6
                            ],
                            linf=[
                                9.57723626271445e-6,
                                1.0480225511866337e-5,
                                1.2817828088262928e-5,
                                4.886962393513272e-5
                            ],
                            alpha=0.1)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.2 * pi" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_rotated.jl"),
                            # Expected errors differ only slightly from elixir_euler_source_terms!
                            l2=[
                                9.32127973957391e-7,
                                8.477824799744325e-7,
                                1.8175286311402784e-6,
                                4.824562453521076e-6
                            ],
                            linf=[
                                9.576898420737834e-6,
                                5.057704352218195e-6,
                                1.635260719945464e-5,
                                4.886978754825577e-5
                            ],
                            alpha=0.2 * pi)
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        let
            t = sol.t[end]
            u_ode = sol.u[end]
            du_ode = similar(u_ode)
            @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
        end
    end

    @trixi_testset "elixir_euler_source_terms_rotated.jl with α = 0.5 * pi" begin
        @test_trixi_include(joinpath(EXAMPLES_DIR,
                                     "elixir_euler_source_terms_rotated.jl"),
                            # Expected errors are exactly the same as in elixir_euler_source_terms!
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
                            alpha=0.5 * pi)
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

@trixi_testset "elixir_euler_source_terms_parallelogram.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_parallelogram.jl"),
                        l2=[
                            1.1167802955144833e-5,
                            1.0805775514153104e-5,
                            1.953188337010932e-5,
                            5.5033856574857146e-5
                        ],
                        linf=[
                            8.297006495561199e-5,
                            8.663281475951301e-5,
                            0.00012264160606778596,
                            0.00041818802502024965
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

@trixi_testset "elixir_euler_source_terms_waving_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_waving_flag.jl"),
                        l2=[
                            2.991891317562739e-5,
                            3.6063177168283174e-5,
                            2.7082941743640572e-5,
                            0.00011414695350996946
                        ],
                        linf=[
                            0.0002437454930492855,
                            0.0003438936171968887,
                            0.00024217622945688078,
                            0.001266380414757684
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
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        l2=[
                            2.063350241405049e-15,
                            1.8571016296925367e-14,
                            3.1769447886391905e-14,
                            1.4104095258528071e-14
                        ],
                        linf=[
                            1.9539925233402755e-14,
                            2.9791447087035294e-13,
                            6.502853810985698e-13,
                            2.7000623958883807e-13
                        ],
                        atol=7.0e-13)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_free_stream.jl with FluxRotated(flux_lax_friedrichs)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        surface_flux=FluxRotated(flux_lax_friedrichs),
                        l2=[
                            2.063350241405049e-15,
                            1.8571016296925367e-14,
                            3.1769447886391905e-14,
                            1.4104095258528071e-14
                        ],
                        linf=[
                            1.9539925233402755e-14,
                            2.9791447087035294e-13,
                            6.502853810985698e-13,
                            2.7000623958883807e-13
                        ],
                        atol=7.0e-13)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[
                            2.259440511901724e-6,
                            2.3188881559075347e-6,
                            2.3188881559568146e-6,
                            6.332786324137878e-6
                        ],
                        linf=[
                            1.4987382622067003e-5,
                            1.918201192063762e-5,
                            1.918201192019353e-5,
                            6.052671713430158e-5
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

@trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.03774907669925568,
                            0.02845190575242045,
                            0.028262802829412605,
                            0.13785915638851698
                        ],
                        linf=[
                            0.3368296929764073,
                            0.27644083771519773,
                            0.27990039685141377,
                            1.1971436487402016
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

@trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            3.69856202e-01,
                            2.35242180e-01,
                            2.41444928e-01,
                            1.28807120e+00
                        ],
                        linf=[
                            1.82786223e+00,
                            1.30452904e+00,
                            1.40347257e+00,
                            6.21791658e+00
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

@trixi_testset "elixir_euler_sedov_blast_wave_sc_subcell.jl (local bounds)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_sedov_blast_wave_sc_subcell.jl"),
                        l2=[
                            0.6403528328480915,
                            0.3068073114438902,
                            0.3140151910019577,
                            1.2977732581465693
                        ],
                        linf=[
                            2.239791987419344,
                            1.5580885989144924,
                            1.5392923786831547,
                            6.2729281824590855
                        ],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Larger values for allowed allocations due to usage of custom
        # integrator which are not *recorded* for the methods from
        # OrdinaryDiffEq.jl
        # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 10000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave_sc_subcell.jl (global bounds)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_sedov_blast_wave_sc_subcell.jl"),
                        positivity_variables_cons=["rho"],
                        positivity_variables_nonlinear=[pressure],
                        local_twosided_variables_cons=[],
                        local_onesided_variables_nonlinear=[],
                        l2=[
                            0.7979084213982606,
                            0.3980284851419719,
                            0.4021949448633982,
                            1.2956482394747346
                        ],
                        linf=[
                            5.477809925838038,
                            3.7793130706228273,
                            3.2838862964081637,
                            6.316943647948965
                        ],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Larger values for allowed allocations due to usage of custom
        # integrator which are not *recorded* for the methods from
        # OrdinaryDiffEq.jl
        # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 10000
    end
end

@trixi_testset "elixir_euler_rayleigh_taylor_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_rayleigh_taylor_instability.jl"),
                        l2=[
                            0.06365630515019809, 0.007166887172039836,
                            0.0028787103533600804, 0.010247678008197966
                        ],
                        linf=[
                            0.47992143569849377, 0.02459548251933757,
                            0.02059810091623976, 0.0319077000843877
                        ],
                        cells_per_dimension=(8, 8),
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

@trixi_testset "elixir_euler_warm_bubble.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_warm_bubble.jl"),
                        l2=[
                            0.00019387402388722496,
                            0.03086514388623955,
                            0.04541427917165,
                            43.892826583444716
                        ],
                        linf=[
                            0.0015942305974430138,
                            0.17449778969139373,
                            0.3729704262394843,
                            307.6706958565337
                        ],
                        cells_per_dimension=(32, 16),
                        tspan=(0.0, 10.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 100
    end
end

@trixi_testset "elixir_eulerpolytropic_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulerpolytropic_convergence.jl"),
                        l2=[
                            0.00166898321776379, 0.00259202637930991,
                            0.0032810744946276406
                        ],
                        linf=[
                            0.010994883201888683, 0.013309526619369905,
                            0.020080326611175536
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

@trixi_testset "elixir_eulerpolytropic_convergence.jl with FluxHLL(min_max_speed_naive)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_eulerpolytropic_convergence.jl"),
                        solver=DGSEM(polydeg = 3,
                                     surface_flux = FluxHLL(min_max_speed_naive),
                                     volume_integral = VolumeIntegralFluxDifferencing(volume_flux)),
                        l2=[
                            0.001668882059653298, 0.002592168188567654,
                            0.0032809503514328307
                        ],
                        linf=[
                            0.01099467966437917, 0.013311978456333584,
                            0.020080117011337606
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

@trixi_testset "elixir_eulerpolytropic_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulerpolytropic_ec.jl"),
                        l2=[
                            0.03647890611450939,
                            0.025284915444045052,
                            0.025340697771609126
                        ],
                        linf=[
                            0.32516731565355583,
                            0.37509762516540046,
                            0.29812843284727336
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

@trixi_testset "elixir_eulerpolytropic_isothermal_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_eulerpolytropic_isothermal_wave.jl"),
                        l2=[
                            0.004998778512795407, 0.004998916021367992,
                            8.991558055435833e-17
                        ],
                        linf=[
                            0.010001103632831354, 0.010051165055185603,
                            7.60697457718599e-16
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

@trixi_testset "elixir_eulerpolytropic_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulerpolytropic_wave.jl"),
                        l2=[
                            0.23642871172548174, 0.2090519382039672,
                            8.778842676292274e-17
                        ],
                        linf=[
                            0.4852276879687425, 0.25327870807625175,
                            5.533921691832115e-16
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

@trixi_testset "elixir_hypdiff_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
                        l2=[0.8799744480157664, 0.8535008397034816, 0.7851383019164209],
                        linf=[1.0771947577311836, 1.9143913544309838, 2.149549109115789],
                        tspan=(0.0, 0.1),
                        coverage_override=(polydeg = 3,)) # Prevent long compile time in CI
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Larger values for allowed allocations due to usage of custom
        # integrator which are not *recorded* for the methods from
        # OrdinaryDiffEq.jl
        # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 15000
    end
end

@trixi_testset "elixir_hypdiff_harmonic_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_hypdiff_harmonic_nonperiodic.jl"),
                        l2=[
                            0.19357947606509474,
                            0.47041398037626814,
                            0.4704139803762686
                        ],
                        linf=[
                            0.35026352556630114,
                            0.8344372248051408,
                            0.8344372248051408
                        ],
                        tspan=(0.0, 0.1),
                        coverage_override=(polydeg = 3,)) # Prevent long compile time in CI
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[0.028905589451357638, 0.006259570019325034,
                            0.005649791156739933, 0.0073272570974805004,
                            0.004890348793116962, 0.00720944138561451,
                            0.0069984328989438115, 0.006729800315219757,
                            0.004318314151888631],
                        linf=[0.17528323378978317, 0.06161030852803388,
                            0.0388335541348234, 0.052906440559080926,
                            0.0380036034027319, 0.04291841215471082,
                            0.03702743958268562, 0.04815794489066357,
                            0.0433064571343779],
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

@trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.0017286908591070864,
                            0.025585037307655684,
                            0.028374244567802766,
                            6.274146767730866e-5
                        ],
                        linf=[
                            0.012973752001194772,
                            0.10829375385832263,
                            0.15832858475438094,
                            0.00018196759554722775
                        ],
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            0.7920927046419308,
                            9.92129670988898e-15,
                            1.0118635033124588e-14,
                            0.7920927046419308
                        ],
                        linf=[
                            2.408429868800133,
                            5.5835419986809516e-14,
                            5.448874313931364e-14,
                            2.4084298688001335
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_ec_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec_shockcapturing.jl"),
                        l2=[0.03641928087745194, 0.04266672246194787,
                            0.042616743034675685,
                            0.025884076832341982,
                            0.16181640309885276, 0.017346521291731105,
                            0.017291600359415987, 0.026856207871456043,
                            0.0007448774124272682],
                        linf=[0.25144155032118376, 0.3288086335996786,
                            0.30532573631664345, 0.20990150465080706,
                            0.9929091025128138, 0.11053858971264774,
                            0.12578085409726314,
                            0.16283334251103732,
                            0.026146463886273865])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_coupled.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_coupled.jl"),
                        l2=[
                            1.0743426980507015e-7, 0.030901698521864966,
                            0.030901698662039206, 0.04370160129981656,
                            8.259193827852516e-8, 0.03090169908364623,
                            0.030901699039770684, 0.04370160128147447,
                            8.735923402748945e-9, 1.0743426996067106e-7,
                            0.03090169852186498, 0.030901698662039206,
                            0.04370160129981657, 8.259193829690747e-8,
                            0.03090169908364624, 0.030901699039770726,
                            0.04370160128147445, 8.73592340076897e-9
                        ],
                        linf=[
                            9.021023431587949e-7, 0.043701454182710486,
                            0.043701458294527366, 0.061803146322536154,
                            9.487023335807976e-7, 0.043701561010342616,
                            0.04370147392153734, 0.06180318786081025,
                            3.430673132525334e-8, 9.02102342825728e-7,
                            0.043701454182710764, 0.043701458294525895,
                            0.06180314632253597, 9.487023254761695e-7,
                            0.04370156101034084, 0.04370147392153745,
                            0.06180318786081015, 3.430672973680963e-8
                        ],
                        coverage_override=(maxiters = 10^5,))

    @testset "analysis_callback(sol) for AnalysisCallbackCoupled" begin
        errors = analysis_callback(sol)
        @test errors.l2≈[
            1.0743426980507015e-7, 0.030901698521864966, 0.030901698662039206,
            0.04370160129981656, 8.259193827852516e-8, 0.03090169908364623,
            0.030901699039770684, 0.04370160128147447, 8.735923402748945e-9,
            1.0743426996067106e-7, 0.03090169852186498, 0.030901698662039206,
            0.04370160129981657, 8.259193829690747e-8, 0.03090169908364624,
            0.030901699039770726, 0.04370160128147445, 8.73592340076897e-9
        ] rtol=1.0e-4
        @test errors.linf≈[
            9.021023431587949e-7, 0.043701454182710486, 0.043701458294527366,
            0.061803146322536154, 9.487023335807976e-7, 0.043701561010342616,
            0.04370147392153734, 0.06180318786081025, 3.430673132525334e-8,
            9.02102342825728e-7, 0.043701454182710764, 0.043701458294525895,
            0.06180314632253597, 9.487023254761695e-7, 0.04370156101034084,
            0.04370147392153745, 0.06180318786081015, 3.430672973680963e-8
        ] rtol=1.0e-4
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
