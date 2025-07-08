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
                        ],)

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
                        ],)

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

@trixi_testset "elixir_advection_meshview.jl with time-dependent CFL" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_meshview.jl"),
                        l2=[
                            8.311947673083206e-6,
                            8.311947673068427e-6
                        ],
                        linf=[
                            6.627000273318195e-5,
                            6.62700027264096e-5
                        ],
                        stepsize_callback=StepsizeCallback(cfl = x -> 1.6))

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
                        linf=[3.438434404412494e-5],)
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
                        restart_file="restart_000000021.h5",)
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
                        restart_file="restart_000000036.h5",)
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

@trixi_testset "elixir_euler_vortex_perk4.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_vortex_perk4.jl"),
                        l2=[
                            0.0001846244731283424,
                            0.00042537910268029285,
                            0.0003724909264689687,
                            0.0026689613797051493
                        ],
                        linf=[
                            0.0025031072787504716,
                            0.009266316022570331,
                            0.009876399281272374,
                            0.0306915591360557
                        ])
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
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 8000
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
                        tspan=(0.0, 0.1),)
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
                        tspan=(0.0, 0.1),)
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

@trixi_testset "elixir_mhd_alfven_wave_er.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_er.jl"),
                        l2=[
                            0.02319878862575278,
                            0.0075015084113375366,
                            0.007501508411337524,
                            0.008501273445248625,
                            0.0052222716672935396,
                            0.00702591840355099,
                            0.0070259184035509695,
                            0.008615547824646918,
                            0.0010828606103854717
                        ],
                        linf=[
                            0.0965835239505668,
                            0.025085249740421506,
                            0.02508524974042145,
                            0.01975892796442895,
                            0.02136629068971274,
                            0.013586601181604374,
                            0.013586601181604374,
                            0.014962260466271568,
                            0.0032435326075120046
                        ])
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 10_000
    end
end

@trixi_testset "elixir_mhd_onion.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_onion.jl"),
                        l2=[0.00614563999392665, 0.04298975803343982,
                            0.009442309044853874, 0.0,
                            0.023466074865980138, 0.0037008480771081663,
                            0.006939946049331198, 0.0, 5.379545284544848e-7],
                        linf=[0.04033992113717799, 0.2507389500590966,
                            0.05597919737542288, 0.0,
                            0.14115256348718308, 0.01995761261479123,
                            0.038667260744994714, 0.0, 3.376777801961409e-6])
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

@trixi_testset "elixir_mhd_orszag_tang_sc_subcell.jl (local * symmetric)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang_sc_subcell.jl"),
                        l2=[
                            0.01971024989875626,
                            0.09104800714369102,
                            0.09850531236459953,
                            0.0,
                            0.11257300398205827,
                            0.0663796508325794,
                            0.1046810844992422,
                            0.0,
                            1.3771070897457708e-7
                        ],
                        linf=[
                            0.06892691571947851,
                            0.2359568430620927,
                            0.27708425716878604,
                            0.0,
                            0.32729450754783485,
                            0.16594293308909247,
                            0.28427225533782474,
                            0.0,
                            1.5760984369383474e-6
                        ],
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 10000
    end
end

@trixi_testset "elixir_mhd_orszag_tang_sc_subcell.jl (local * jump)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang_sc_subcell.jl"),
                        l2=[
                            0.019710787852084945,
                            0.09104739316084506,
                            0.09850451818593346,
                            0.0,
                            0.11257089275762928,
                            0.0663755234418436,
                            0.10468586115056747,
                            0.0,
                            4.200881361783599e-6
                        ],
                        linf=[
                            0.06893188693406871,
                            0.23594610243501996,
                            0.2770924621975269,
                            0.0,
                            0.32731120349573106,
                            0.1659395971443428,
                            0.2842678645407109,
                            0.0,
                            2.6014507178710646e-5
                        ],
                        surface_flux=(flux_lax_friedrichs,
                                      flux_nonconservative_powell_local_jump),
                        volume_flux=(flux_central,
                                     flux_nonconservative_powell_local_jump),
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 10000
    end
end

@trixi_testset "elixir_mhd_coupled.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_coupled.jl"),
                        l2=[
                            1.0743426976677776e-7,
                            5.941703122781545e-8,
                            6.373264854058786e-8,
                            1.0327320202980158e-7,
                            8.259193826511926e-8,
                            8.377839796183567e-8,
                            7.469434303577898e-8,
                            1.0770585130793933e-7,
                            8.735923402823923e-9,
                            1.0743426990741475e-7,
                            5.941703121622708e-8,
                            6.373264853185012e-8,
                            1.0327320202884373e-7,
                            8.259193828324533e-8,
                            8.377839796046157e-8,
                            7.469434302767398e-8,
                            1.077058513088068e-7,
                            8.735923400740853e-9
                        ],
                        linf=[
                            9.021023420485719e-7,
                            5.540360292766167e-7,
                            8.97403747285308e-7,
                            9.962467816537757e-7,
                            9.48702334468976e-7,
                            1.4284730157632097e-6,
                            5.317911039304235e-7,
                            9.92786089865083e-7,
                            3.4306731372516224e-8,
                            9.021023412714158e-7,
                            5.540360226014007e-7,
                            8.974037428166604e-7,
                            9.962467838325884e-7,
                            9.487023256982141e-7,
                            1.4284730160962766e-6,
                            5.317911003777098e-7,
                            9.92786092363085e-7,
                            3.430672968714232e-8
                        ],)

    @testset "analysis_callback(sol) for AnalysisCallbackCoupled" begin
        errors = analysis_callback(sol)
        @test errors.l2≈[
            1.0743426976677776e-7,
            5.941703122781545e-8,
            6.373264854058786e-8,
            1.0327320202980158e-7,
            8.259193826511926e-8,
            8.377839796183567e-8,
            7.469434303577898e-8,
            1.0770585130793933e-7,
            8.735923402823923e-9,
            1.0743426990741475e-7,
            5.941703121622708e-8,
            6.373264853185012e-8,
            1.0327320202884373e-7,
            8.259193828324533e-8,
            8.377839796046157e-8,
            7.469434302767398e-8,
            1.077058513088068e-7,
            8.735923400740853e-9
        ] rtol=1.0e-4
        @test errors.linf≈[
            9.021023420485719e-7,
            5.540360292766167e-7,
            8.97403747285308e-7,
            9.962467816537757e-7,
            9.48702334468976e-7,
            1.4284730157632097e-6,
            5.317911039304235e-7,
            9.92786089865083e-7,
            3.4306731372516224e-8,
            9.021023412714158e-7,
            5.540360226014007e-7,
            8.974037428166604e-7,
            9.962467838325884e-7,
            9.487023256982141e-7,
            1.4284730160962766e-6,
            5.317911003777098e-7,
            9.92786092363085e-7,
            3.430672968714232e-8
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

@trixi_testset "elixir_lbm_lid_driven_cavity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_lid_driven_cavity.jl"),
                        l2=[
                            0.0013650620243296592,
                            0.00022198751341720896,
                            0.0012598874493852138,
                            0.0003717179135584138,
                            0.0004378131417115368,
                            0.0003981707758995024,
                            0.00025217328296435736,
                            0.00026487031088613346,
                            0.0004424433618470548
                        ],
                        linf=[
                            0.024202160934419875,
                            0.011909887052061488,
                            0.021787515301598115,
                            0.03618036838142735,
                            0.008017773116953682,
                            0.0068482058999433,
                            0.010286155761527443,
                            0.009919734282811003,
                            0.05568155678921127
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

@trixi_testset "elixir_lbm_eulerpolytropic_coupled.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_lbm_eulerpolytropic_coupled.jl"),
                        l2=[
                            0.004425408662988481,
                            0.004450324455480091,
                            6.443442487292444e-17,
                            0.0013646410236054789,
                            0.000492124768468392,
                            0.00035879680384107377,
                            0.0004921247684683822,
                            0.0003411602559013719,
                            8.969920096027091e-5,
                            8.969920096027404e-5,
                            0.00034116025590136945,
                            0.001968499073873568
                        ],
                        linf=[
                            0.009769926457488198,
                            0.009821015729172138,
                            3.313984464407251e-16,
                            0.003072464362545338,
                            0.001104208150516095,
                            0.000791310479149987,
                            0.0011042081505159979,
                            0.000768116090636338,
                            0.0001978276197874898,
                            0.00019782761978750715,
                            0.0007681160906363102,
                            0.0044168326020643245
                        ])

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
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
