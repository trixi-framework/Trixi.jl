@testsnippet StructuredMesh2D begin
    EXAMPLES_DIR = joinpath(examples_dir(), "structured_2d_dgsem")
end

@testitem "StructuredMesh2D: elixir_advection_basic.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_float32.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    # Expected errors are taken from elixir_advection_basic.jl
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_float32.jl"),
                        # Expected errors are taken from elixir_advection_basic.jl
                        l2=[Float32(8.311947673061856e-6)],
                        linf=[Float32(6.627000273229378e-5)],
                        RealT_for_test_tolerances=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_coupled.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
                        ])

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
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end

    # Test plotdata construction for coupled semidiscretization
    @test_nowarn pd = PlotData2D(sol)
end

@testitem "StructuredMesh2D: elixir_advection_meshview.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_meshview.jl"),
                        l2=[
                            8.311947673083206e-6,
                            8.311947673068427e-6
                        ],
                        linf=[
                            6.627000273318195e-5,
                            6.62700027264096e-5
                        ])

    @testset "analysis_callback(sol) for AnalysisCallbackCoupled" begin
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end
end

@testitem "StructuredMesh2D: elixir_advection_meshview.jl with time-dependent CFL" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end
end

@testitem "StructuredMesh2D: elixir_advection_extended.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[4.220397559713772e-6],
                        linf=[3.477948874874848e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_extended.jl with polydeg=4" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_extended.jl"),
                        l2=[5.32996976442737e-7],
                        linf=[4.1344662966569246e-6],
                        atol=1e-12, # required to make CI tests pass on macOS
                        cells_per_dimension=(16, 23),
                        polydeg=4,
                        cfl=1.4)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_rotated.jl with α = 0.0" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
                        # Expected errors are exactly the same as in elixir_advection_basic!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5],
                        alpha=0.0)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_rotated.jl with α = 0.1" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
                        # Expected errors differ only slightly from elixir_advection_basic!
                        l2=[8.3122750550501e-6],
                        linf=[6.626802581322089e-5],
                        alpha=0.1)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_rotated.jl with α = 0.5 * pi" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_rotated.jl"),
                        # Expected errors are exactly the same as in elixir_advection_basic!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5],
                        alpha=0.5 * pi)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_parallelogram.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_parallelogram.jl"),
                        # Expected errors are exactly the same as in elixir_advection_basic!
                        l2=[8.311947673061856e-6],
                        linf=[6.627000273229378e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_waving_flag.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_waving_flag.jl"),
                        l2=[0.00018553859900545866],
                        linf=[0.0016167719118129753])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_free_stream.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream.jl"),
                        l2=[6.8925194184204476e-15],
                        linf=[9.903189379656396e-14])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_nonperiodic.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonperiodic.jl"),
                        l2=[0.00025552740731641223],
                        linf=[0.007252625722805939])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_restart.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[4.219208035582454e-6],
                        linf=[3.438434404412494e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_restart.jl with waving flag mesh" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[0.00016265538265929818],
                        linf=[0.0015194252169410394],
                        rtol=5.0e-5, # Higher tolerance to make tests pass in CI (in particular with macOS)
                        elixir_file="elixir_advection_waving_flag.jl",
                        restart_file="restart_000000021.h5")
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_advection_restart.jl with free stream mesh" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[7.841217436552029e-15],
                        linf=[1.0857981180834031e-13],
                        elixir_file="elixir_advection_free_stream.jl",
                        restart_file="restart_000000036.h5")
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_convergence_implicit_sparse_jacobian.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_convergence_implicit_sparse_jacobian.jl"),
                        tspan=(0.0, 1.0),
                        l2=[
                            0.0025545032994393493,
                            0.0025848892135096136,
                            0.002585815262287367,
                            0.0031668773337869584
                        ],
                        linf=[
                            0.010367159504626189,
                            0.009326212633131492,
                            0.008372785091578683,
                            0.011242647117379434
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_convergence_implicit_sparse_jacobian.jl with flux_ranocha" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_convergence_implicit_sparse_jacobian.jl"),
                        solver=DGSEM(polydeg = 3, surface_flux = surface_flux,
                                     volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha)),
                        tspan=(0.0, 1.0),
                        l2=[
                            0.002488034310310255,
                            0.002537347299714133,
                            0.002533529761212216,
                            0.0030150881617191675
                        ],
                        linf=[
                            0.005844630331331979,
                            0.005507186931414498,
                            0.005377359689946237,
                            0.00631648929531492
                        ],
                        # Relax error tols to avoid stochastic CI failures
                        atol=1e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_eulermulti_convergence_ec.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_eulermulti_blastwave_ec.jl with boundary_condition_slip_wall" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_blastwave_ec.jl"),
                        l2=[
                            0.005884923780993405,
                            0.005815148890903912,
                            0.023438850211096113,
                            0.006254106697017372
                        ],
                        linf=[
                            0.24483047700349253,
                            0.13364458078315503,
                            0.38469398740194993,
                            0.1024647566986494
                        ],
                        tspan=(0.0, 0.001))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms_rotated.jl with α = 0.0" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms_rotated.jl with α = 0.1" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms_rotated.jl with α = 0.2 * pi" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms_rotated.jl with α = 0.5 * pi" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms_parallelogram.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms_waving_flag.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_free_stream.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
# `StepsizeCallback` (CFL-Condition) and less diffusion.
@testitem "StructuredMesh2D: elixir_euler_free_stream.jl with FluxRotated(FluxLaxFriedrichs(max_abs_speed_naive))" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        surface_flux=FluxRotated(FluxLaxFriedrichs(max_abs_speed_naive)),
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms_nonperiodic.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_source_terms_nonperiodic_fvO2.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic_fvO2.jl"),
                        l2=[
                            0.0027535201954222072,
                            0.0017808463145373606,
                            0.0017808463145373874,
                            0.005589356782700206
                        ],
                        linf=[
                            0.010801198634897702,
                            0.00787469718577416,
                            0.007874697185775936,
                            0.02417877751394304
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_vortex_perk4.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 8000)
end

@testitem "StructuredMesh2D: elixir_euler_ec.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_sedov.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_sedov_blast_wave_sc_subcell.jl (local bounds)" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_sedov_blast_wave_sc_subcell.jl"),
                        l2=[
                            0.6401790809469766,
                            0.30665376287777196,
                            0.31392947418333217,
                            1.297784156450693
                        ],
                        linf=[
                            2.247645842347004,
                            1.554935338577008,
                            1.5439591073963144,
                            6.272642656315988
                        ],
                        tspan=(0.0, 0.5))
    limiter = semi.solver.volume_integral.limiter
    deviations = collect(values(limiter.cache.idp_bounds_delta_global))
    @test all(isfinite, deviations)
    @test maximum(deviations) <= 7.0e-13

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 10000)
end

@testitem "StructuredMesh2D: elixir_euler_sedov_blast_wave_sc_subcell.jl (global bounds)" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_sedov_blast_wave_sc_subcell.jl"),
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
    limiter = semi.solver.volume_integral.limiter
    deviations = collect(values(limiter.cache.idp_bounds_delta_global))
    @test all(isfinite, deviations)
    @test maximum(deviations) <= 1.0e-13

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 10000)
end

@testitem "StructuredMesh2D: elixir_euler_rayleigh_taylor_instability.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_richtmyer_meshkov.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_richtmyer_meshkov.jl"),
                        l2=[
                            0.11325554126117648,
                            3.846072397992506e-14,
                            0.30067515354593266,
                            0.4451325203339372
                        ],
                        linf=[
                            0.4630182829483884,
                            3.795750685486191e-13,
                            1.265147868668324,
                            1.8837528418813672
                        ],
                        adaptive=false, dt=1e-2,
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_euler_warm_bubble.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 100)
end

@testitem "StructuredMesh2D: elixir_euler_peng_robinson_transcritical_mixing" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_peng_robinson_transcritical_mixing.jl"),
                        tspan=(0.0, 0.0003),
                        # note that errors are large because the solution values are of the order 1e5-1e7
                        l2=[
                            0.8907552376416852,
                            274.6262332037992,
                            129.95629990639333,
                            94420.33529773205
                        ],
                        linf=[
                            6.617401501819359,
                            732.0947275447616,
                            403.74606195408825,
                            584504.7663076259
                        ])

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_eulerpolytropic_convergence.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_eulerpolytropic_convergence.jl with FluxHLL(min_max_speed_naive)" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_eulerpolytropic_convergence.jl sparsity detection" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_eulerpolytropic_convergence.jl"),
                        surface_flux=flux_lax_friedrichs,
                        tspan=(0.0, 0.0))

    import SparseConnectivityTracer: TracerSparsityDetector,
                                     jacobian_eltype,
                                     jacobian_sparsity

    jac_detector = TracerSparsityDetector()
    # We need to construct the semidiscretization with the correct
    # sparsity-detection ready datatype, which is retrieved here
    jac_eltype = jacobian_eltype(real(solver), jac_detector)

    # Semidiscretization for sparsity pattern detection
    semi_jac_type = SemidiscretizationHyperbolic(mesh, equations,
                                                 initial_condition,
                                                 solver;
                                                 boundary_conditions = boundary_condition_periodic,
                                                 source_terms = source_terms_convergence_test,
                                                 uEltype = jac_eltype) # Need to supply Jacobian element type

    # Call `semidiscretize` to create the ODE problem to have access to the
    # initial condition based on which the sparsity pattern is computed
    ode_jac_type = semidiscretize(semi_jac_type, tspan)
    u0_ode = ode_jac_type.u0
    du_ode = similar(u0_ode)

    ###############################################################################
    ### Compute the Jacobian sparsity pattern ###

    # Wrap the `Trixi.rhs_hyperbolic!` function to match the signature `f!(du, u)`, see
    # https://adrianhill.de/SparseConnectivityTracer.jl/stable/user/api/#ADTypes.jacobian_sparsity
    rhs_wrapped! = function (du_ode, u0_ode)
        Trixi.rhs_hyperbolic!(du_ode, u0_ode, semi_jac_type, tspan[1])
    end

    @test_nowarn jacobian_sparsity(rhs_wrapped!, du_ode, u0_ode, jac_detector)
end

@testitem "StructuredMesh2D: elixir_eulerpolytropic_ec.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_eulerpolytropic_isothermal_wave.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_eulerpolytropic_wave.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_hypdiff_nonperiodic.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_hypdiff_nonperiodic.jl"),
                        l2=[0.8799744480157664, 0.8535008397034816, 0.7851383019164209],
                        linf=[1.0771947577311836, 1.9143913544309838, 2.149549109115789],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15000)
end

@testitem "StructuredMesh2D: elixir_hypdiff_harmonic_nonperiodic.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_mhd_ec.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
                        l2=[0.04933104425405548, 0.06115966545280778,
                            0.06095935716042677, 0.031535344983242605,
                            0.23174010718715246, 0.024756245423285066,
                            0.02454330651313403, 0.03540888614254055,
                            0.003198990190614683],
                        linf=[0.23699011258872849, 0.3010922751593751,
                            0.39031674641843217, 0.22168707174529842,
                            0.9835232760382162, 0.1229163010648412,
                            0.13977781294096125, 0.17663469078477922,
                            0.08083790569368657],
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_mhd_alfven_wave.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[0.028664696412554345, 0.006117836928758066,
                            0.005489176505206269, 0.007149881621841527,
                            0.004028600057920484, 0.004710199728864059,
                            0.004569284341705584, 0.006639242089875928,
                            0.0029931853637171495],
                        linf=[0.14889030979516216, 0.03977580497666612,
                            0.042091712890758894, 0.051229697226991694,
                            0.02558169128224097, 0.020070800501665875,
                            0.025298017941231765, 0.04759102530251386,
                            0.01698911973096513],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_mhd_onion.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_onion.jl"),
                        l2=[0.00614563999392665, 0.04298975803343982,
                            0.009442309044853874, 0.0,
                            0.023466074865980138, 0.003700848245122418,
                            0.0069399462910863146, 0.0, 2.6526292145637484e-6],
                        linf=[0.04034000344526767, 0.25073951149407875,
                            0.05597857594719358, 0.0,
                            0.14115800038105464, 0.019956735193906228,
                            0.0386738912652137, 0.0, 2.1686817835052344e-5])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_mhd_ec_shockcapturing.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec_shockcapturing.jl"),
                        l2=[0.03641112966661276, 0.04266066899161558,
                            0.04260994412980411,
                            0.025872361672528356,
                            0.16175824450493093, 0.01735408672130778,
                            0.01730164282204929, 0.026841409075862483,
                            0.0012865132289044427],
                        linf=[0.2491951481802378, 0.3288786693296565,
                            0.3005271580981498, 0.2118469995362261,
                            0.9662026647020632, 0.10585841999884227,
                            0.12590330776574343,
                            0.16026062839794064,
                            0.028354117953987793])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_mhd_orszag_tang_sc_subcell.jl (local * symmetric)" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
                            3.3691335574770936e-6
                        ],
                        linf=[
                            0.06892663750784545,
                            0.23595702378539926,
                            0.2770852926790989,
                            0.0,
                            0.3273010179518012,
                            0.16594172124166834,
                            0.2842715169518112,
                            0.0,
                            2.8911656850049133e-5
                        ],
                        tspan=(0.0, 0.025))
    limiter = semi.solver.volume_integral.limiter
    deviations = collect(values(limiter.cache.idp_bounds_delta_global))
    @test all(isfinite, deviations)
    @test maximum(deviations) <= 1.0e-13

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 10000)
end

@testitem "StructuredMesh2D: elixir_mhd_orszag_tang_sc_subcell.jl (local * jump)" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang_sc_subcell.jl"),
                        l2=[
                            0.019710249751507508,
                            0.09104800760742554,
                            0.09850531236986743,
                            0.0,
                            0.1125730051496663,
                            0.06637965089746012,
                            0.10468108506207952,
                            0.0,
                            2.7355684063483483e-6
                        ],
                        linf=[
                            0.0689267300108618,
                            0.2359580791896736,
                            0.2770849766159345,
                            0.0,
                            0.3273050470240757,
                            0.1659419573015099,
                            0.28427211213754544,
                            0.0,
                            2.4706566865940354e-5
                        ],
                        # Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
                        # `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
                        # In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
                        # Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
                        # To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
                        # We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
                        # `StepsizeCallback` (CFL-Condition) and less diffusion.
                        surface_flux=(FluxLaxFriedrichs(max_abs_speed_naive),
                                      flux_nonconservative_powell_local_jump),
                        volume_flux=(flux_central,
                                     flux_nonconservative_powell_local_jump),
                        tspan=(0.0, 0.025))
    limiter = semi.solver.volume_integral.limiter
    deviations = collect(values(limiter.cache.idp_bounds_delta_global))
    @test all(isfinite, deviations)
    @test maximum(deviations) <= 1.0e-13

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 10000)
end

@testitem "StructuredMesh2D: elixir_mhdmultiion_ec.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.001213163388290671,
                            0.001213088378349994,
                            0.0015755001836252462,
                            0.0013582923009927254,
                            0.0020117645260758414,
                            0.002011172963281366,
                            4.8097671254871054e-5,
                            0.017144538704574448,
                            0.002725517676213416,
                            0.002881115950314307,
                            0.002879122244461666,
                            0.00019209436068007779,
                            0.011925525689473708,
                            1.2847538494838258e-6
                        ],
                        linf=[
                            0.0978299330911645,
                            0.09693277394423094,
                            0.1343132092378978,
                            0.05311489239555983,
                            0.09355072731834056,
                            0.09355996145995278,
                            0.004795104818302111,
                            0.8723959821253295,
                            0.10974818170313227,
                            0.15983724793238469,
                            0.13675344214792837,
                            0.019081506400089376,
                            0.7128069726304465,
                            0.0002655093808709079
                        ],
                        tspan=(0.0, 0.002))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_mhdmultiion_ec.jl with local Lax-Friedrichs at the surface" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.001103121194053902,
                            0.0011031409545128022,
                            0.0014691743776654273,
                            0.0013355107526990084,
                            0.0019050117958305927,
                            0.001903829076090151,
                            3.6572020823936584e-5,
                            0.016664169283889124,
                            0.0026895584922699152,
                            0.0027876396404945794,
                            0.002784300002272668,
                            0.00014627107562550072,
                            0.011549903926786112,
                            3.352631893758921e-5
                        ],
                        linf=[
                            0.06948689173795997,
                            0.06937757460089966,
                            0.11513858545843825,
                            0.04527021012505911,
                            0.082045761192407,
                            0.08205460646907536,
                            0.0028240886950803836,
                            0.7319352744042775,
                            0.09235868357225929,
                            0.1612912038455094,
                            0.1243984432903087,
                            0.011236732132802112,
                            0.5923283432735742,
                            0.005783424579094766
                        ],
                        tspan=(0.0, 0.002),
                        surface_flux=(FluxLaxFriedrichs(max_abs_speed_naive),
                                      flux_nonconservative_central))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_mhdmultiion_convergence_twospecies.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhdmultiion_convergence_twospecies.jl"),
                        l2=[
                            0.0010503125209059497,
                            0.0012534373020343082,
                            0.0002811481549151607,
                            0.001134699506017358,
                            0.004308614124900076,
                            0.004035396191750047,
                            0.00016305325832953208,
                            0.009626514104735253,
                            0.0016373697739567035,
                            0.008306218636498714,
                            0.008422985804717287,
                            0.00026937713085479467,
                            0.017967149724963348,
                            0.0012916482409408318
                        ],
                        linf=[
                            0.004673805602367853,
                            0.00857241232460726,
                            0.0016219747363969972,
                            0.00589494836970883,
                            0.02322082052541652,
                            0.020155703478578202,
                            0.0007195183293943891,
                            0.05125865459457701,
                            0.010507512452583745,
                            0.03912569507004138,
                            0.033336249093876225,
                            0.001450791062907797,
                            0.1022349242163818,
                            0.0059079203615904135
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_mhd_coupled.jl" setup=[Setup, StructuredMesh2D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_coupled.jl"),
                        l2=[
                            1.0910422674017391e-7,
                            5.949506169238524e-8,
                            6.381077792964088e-8,
                            1.0332534335510831e-7,
                            7.446298701601602e-8,
                            6.82341025366468e-8,
                            6.847267564928107e-8,
                            1.0770420753774933e-7,
                            3.259113678757843e-8,
                            1.0910422687503122e-7,
                            5.9495061683399424e-8,
                            6.381077791987118e-8,
                            1.0332534335845108e-7,
                            7.446298701485028e-8,
                            6.823410252498349e-8,
                            6.847267565099689e-8,
                            1.0770420753888783e-7,
                            3.259113677895644e-8
                        ],
                        linf=[
                            9.228681541184613e-7,
                            5.623746595490031e-7,
                            8.55291005422143e-7,
                            9.967290961060415e-7,
                            5.965679433783677e-7,
                            5.940604139720307e-7,
                            5.757758608782026e-7,
                            9.932069444662517e-7,
                            2.8771475120192863e-7,
                            9.228681583373088e-7,
                            5.623746533872653e-7,
                            8.552910063658326e-7,
                            9.967290983403654e-7,
                            5.965679429342785e-7,
                            5.940604121956738e-7,
                            5.757758602120688e-7,
                            9.932069467560867e-7,
                            2.877147484184855e-7
                        ])

    @testset "analysis_callback(sol) for AnalysisCallbackCoupled" begin
        errors = analysis_callback(sol)
        @test errors.l2≈[
            1.0910422674017391e-7,
            5.949506169238524e-8,
            6.381077792964088e-8,
            1.0332534335510831e-7,
            7.446298701601602e-8,
            6.82341025366468e-8,
            6.847267564928107e-8,
            1.0770420753774933e-7,
            3.259113678757843e-8,
            1.0910422687503122e-7,
            5.9495061683399424e-8,
            6.381077791987118e-8,
            1.0332534335845108e-7,
            7.446298701485028e-8,
            6.823410252498349e-8,
            6.847267565099689e-8,
            1.0770420753888783e-7,
            3.259113677895644e-8
        ] rtol=1.0e-4
        @test errors.linf≈[
            9.228681541184613e-7,
            5.623746595490031e-7,
            8.55291005422143e-7,
            9.967290961060415e-7,
            5.965679433783677e-7,
            5.940604139720307e-7,
            5.757758608782026e-7,
            9.932069444662517e-7,
            2.8771475120192863e-7,
            9.228681583373088e-7,
            5.623746533872653e-7,
            8.552910063658326e-7,
            9.967290983403654e-7,
            5.965679429342785e-7,
            5.940604121956738e-7,
            5.757758602120688e-7,
            9.932069467560867e-7,
            2.877147484184855e-7
        ] rtol=1.0e-4
        # Ensure that we do not have excessive memory allocations
        # (e.g., from type instabilities)
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end
end

@testitem "StructuredMesh2D: elixir_lbm_lid_driven_cavity.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh2D: elixir_lbm_eulerpolytropic_coupled.jl" setup=[
    Setup,
    StructuredMesh2D
] tags=[:structured] begin
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
        @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    end
end
