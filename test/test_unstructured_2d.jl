@testsnippet UnstructuredMesh2D begin
    using Adapt
    EXAMPLES_DIR = joinpath(examples_dir(), "unstructured_2d_dgsem")
end

@testitem "UnstructuredMesh2D: elixir_euler_periodic.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic.jl"),
                        l2=[
                            0.00010992161458946449, 0.00013037957831794187,
                            0.0001303795783182231, 0.00029937279106433015
                        ],
                        linf=[
                            0.006407280909098478, 0.009836067162309448,
                            0.009836067162292128, 0.021903519372712843
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_euler_periodic.jl (O2 inner reconstruction)" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic.jl"),
                        solver=DGSEM(polydeg = 6, surface_flux = flux_hll,
                                     volume_integral = VolumeIntegralPureLGLFiniteVolumeO2(LobattoLegendreBasis(6),
                                                                                           volume_flux_fv = flux_hll,
                                                                                           reconstruction_mode = reconstruction_O2_inner,
                                                                                           slope_limiter = vanleer)),
                        l2=[
                            0.005880232650480498, 0.0040529619297677115,
                            0.004052961929767421, 0.010484942040225525
                        ],
                        linf=[
                            0.01573598342343474, 0.011413826186375431,
                            0.011413826186384313, 0.028639280744286122
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_euler_free_stream.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        l2=[
                            3.3937365073416665e-14, 2.44759188939065e-13,
                            1.4585198700082895e-13, 4.716940764877479e-13
                        ],
                        linf=[
                            7.774003663030271e-12, 9.183176441496244e-11,
                            4.5685344396417804e-11, 1.0534506600379245e-10
                        ],
                        tspan=(0.0, 0.1),
                        atol=3.0e-13)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_euler_wall_bc.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_wall_bc.jl"),
                        l2=[
                            0.040189107976346644,
                            0.04256154998030852,
                            0.03734120743842209,
                            0.10057425897733507
                        ],
                        linf=[
                            0.24455374304626365,
                            0.2970686406973577,
                            0.29339040847600434,
                            0.5915610037764794
                        ],
                        tspan=(0.0, 0.25),
                        surface_flux=FluxHLL(min_max_speed_naive))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_euler_basic.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    using Trixi: default_example_unstructured
    @test_trixi_include(default_example_unstructured(),
                        l2=[
                            0.0007213418215265047,
                            0.0006752337675043779,
                            0.0006437485997536973,
                            0.0014782883071363362
                        ],
                        linf=[
                            0.004301288971032324,
                            0.005243995459478956,
                            0.004685630332338153,
                            0.01750217718347713
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_euler_restart.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_restart.jl"),
                        l2=[
                            0.0007213418215265047,
                            0.0006752337675043779,
                            0.0006437485997536973,
                            0.0014782883071363362
                        ],
                        linf=[
                            0.004301288971032324,
                            0.005243995459478956,
                            0.004685630332338153,
                            0.01750217718347713
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_euler_ec.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.06594600495903137,
                            0.10803914821786433,
                            0.10805946357846291,
                            0.1738171782368222
                        ],
                        linf=[
                            0.31880214280781305,
                            0.3468488554333352,
                            0.34592958184413264,
                            0.784555926860546
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_advection_basic.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.00018729339078205488],
                        linf=[0.0018997287705734278])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_euler_sedov.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            2.19945600e-01,
                            1.71050453e-01,
                            1.71050453e-01,
                            1.21719195e+00
                        ],
                        linf=[
                            7.44218635e-01,
                            7.02887039e-01,
                            7.02887039e-01,
                            6.11732719e+00
                        ],
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_euler_time_series.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_time_series.jl"),
                        l2=[
                            6.984024099236519e-5,
                            6.289022520363763e-5,
                            6.550951878107466e-5,
                            0.00016222767700879948
                        ],
                        linf=[
                            0.0005367823248620951,
                            0.000671293180158461,
                            0.0005656680962440319,
                            0.0013910024779804075
                        ],
                        tspan=(0.0, 0.2))
    # Extra test that the `TimeSeries` callback creates reasonable data
    point_data_1 = time_series.affect!.point_data[1]
    @test all(isapprox.(point_data_1[1:4],
                        [1.9546882708551676, 1.9547149531788077,
                            1.9547142161310154, 3.821066781119142]))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_acoustics_gauss_wall.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_acoustics_gauss_wall.jl"),
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_mhd_ec.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
                        l2=[0.06415170466136184, 0.12085408678329461,
                            0.12085328678029783, 0.07742277881082815,
                            0.16229125591404286, 0.040406655277589236,
                            0.04040712063824585, 0.05733786787817643,
                            0.002648044431186657],
                        linf=[0.14045766664607662, 0.32623576679739064,
                            0.34476362755121487,
                            0.22334073507496888,
                            0.43700327654947513, 0.15484260279789186,
                            0.15436212102384317, 0.1770287052552837,
                            0.01791098053009698],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_mhd_alfven_wave.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            5.096781009669014e-5,
                            5.572768183478255e-6,
                            5.572768182784067e-6,
                            7.89961702644077e-6,
                            3.818117216433364e-6,
                            3.7187227000871942e-6,
                            3.7187227001967664e-6,
                            5.905662978494377e-6,
                            3.4613006267225358e-6
                        ],
                        linf=[
                            0.00034828857054070017,
                            5.7507509292745174e-5,
                            5.7507509122922684e-5,
                            5.393301837056108e-5,
                            3.208364835805e-5,
                            4.649972580106354e-5,
                            4.6499725794513225e-5,
                            6.632941531577174e-5,
                            2.5164965312816113e-5
                        ],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: elixir_mhd_onion.jl" setup=[Setup, UnstructuredMesh2D] tags=[:unstructured_dgmulti] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_onion.jl"),
                        l2=[
                            0.006145566821936454,
                            0.04298942319307415,
                            0.009442052851913185,
                            0.0,
                            0.023467484843028026,
                            0.00370358905176887,
                            0.0069419050109037535,
                            0.0,
                            0.0001367246545383922
                        ],
                        linf=[
                            0.040343698183574084,
                            0.2507806983620838,
                            0.05618538436557681,
                            0.0,
                            0.1413893963817512,
                            0.020120977632335113,
                            0.038666529439023045,
                            0.0,
                            0.0027742963554038833
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

# TODO: FD; for now put the unstructured tests for the 2D FDSBP here.
@testitem "UnstructuredMesh2D: FDSBP (central): elixir_advection_basic.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    using Trixi: examples_dir
    @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_fdsbp",
                                 "elixir_advection_basic.jl"),
                        l2=[0.0001105211407319266],
                        linf=[0.0004199363734466166])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: FDSBP (central): elixir_euler_ec.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    using Trixi: examples_dir
    @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_fdsbp",
                                 "elixir_euler_ec.jl"),
                        l2=[0.09056730335512135,
                            0.07557730593078431,
                            0.07557331604286137,
                            0.23768999194184112],
                        linf=[0.37265499941374935,
                            0.25381409473468786,
                            0.2584902123868289,
                            0.9037236407898466],
                        tspan=(0.0, 0.2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: FDSBP (central): elixir_euler_source_terms.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    using Trixi: examples_dir
    @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_fdsbp",
                                 "elixir_euler_source_terms.jl"),
                        l2=[8.155544666380138e-5,
                            0.0001477863788446318,
                            0.00014778637884460072,
                            0.00045584189984542687],
                        linf=[0.0002670775876922882,
                            0.0005683064706873964,
                            0.0005683064706762941,
                            0.0017770812025146299],
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: FDSBP (central): elixir_euler_free_stream.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    using Trixi: examples_dir
    @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_fdsbp",
                                 "elixir_euler_free_stream.jl"),
                        l2=[5.4329175009362306e-14,
                            1.0066867437607972e-13,
                            6.889210012578449e-14,
                            1.568290814572709e-13],
                        linf=[5.6139981552405516e-11,
                            2.842849566864203e-11,
                            1.8290174930157832e-11,
                            4.61017890529547e-11],
                        tspan=(0.0, 0.1),
                        atol=1.0e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: FDSBP (upwind): elixir_euler_source_terms_upwind.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    using Trixi: examples_dir
    @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_fdsbp",
                                 "elixir_euler_source_terms_upwind.jl"),
                        l2=[4.085391175504837e-5,
                            7.19179253772227e-5,
                            7.191792537723135e-5,
                            0.0002177522206115571],
                        linf=[0.0004054489124620808,
                            0.0006164432358217731,
                            0.0006164432358186644,
                            0.001363103391379461],
                        tspan=(0.0, 0.05),
                        atol=2.0e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: FDSBP (upwind): elixir_euler_source_terms_upwind.jl with LF splitting" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    using Trixi: examples_dir
    @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_fdsbp",
                                 "elixir_euler_source_terms_upwind.jl"),
                        l2=[
                            3.8300274213823844e-5,
                            5.295847308063744e-5,
                            5.295847308066259e-5,
                            0.00017564760044077823
                        ],
                        linf=[
                            0.0001881071653406785,
                            0.00037941874324110003,
                            0.0003794187432419882,
                            0.0009632958467822306
                        ],
                        tspan=(0.0, 0.025),
                        flux_splitting=splitting_lax_friedrichs,
                        atol=2e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: FDSBP (upwind): elixir_euler_free_stream_upwind.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    using Trixi: examples_dir
    @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_fdsbp",
                                 "elixir_euler_free_stream_upwind.jl"),
                        l2=[3.2114065566681054e-14,
                            2.132488788134846e-14,
                            2.106144937311659e-14,
                            8.609642264224197e-13],
                        linf=[3.354871935812298e-11,
                            7.006478730531285e-12,
                            1.148153794261475e-11,
                            7.461231632532872e-10],
                        tspan=(0.0, 0.05),
                        atol=1.0e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "UnstructuredMesh2D: FDSBP (upwind): elixir_euler_free_stream_upwind_float32.jl" setup=[
    Setup,
    UnstructuredMesh2D
] tags=[:unstructured_dgmulti] begin
    using Trixi: examples_dir
    @test_trixi_include(joinpath(examples_dir(), "unstructured_2d_fdsbp",
                                 "elixir_euler_free_stream_upwind_float32.jl"),
                        l2=[0, 0, 0, 0],
                        linf=[0, 0, 0, 0],
                        tspan=(0.0f0, 0.05f0),
                        # The `Float32` round-off errors depend on the hardware
                        # (e.g., FMA availability) and the adaptive step sequence;
                        # the `linf` error of `rho_e_total ≈ 25` varies between
                        # about `6.0f-4` and `1.8f-3`.
                        atol=5.0f-3)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
