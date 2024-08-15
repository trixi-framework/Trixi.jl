module TestExamplesUnstructuredMesh2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "unstructured_2d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "UnstructuredMesh2D" begin
#! format: noindent

@trixi_testset "elixir_euler_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_periodic.jl"),
                        l2=[
                            0.0001099216141882387, 0.0001303795774982892,
                            0.00013037957749794242, 0.0002993727892598759,
                        ],
                        linf=[
                            0.006407280810928562, 0.009836067015418948,
                            0.009836067015398076, 0.021903519038095176,
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
                            3.3937365073416665e-14, 2.44759188939065e-13,
                            1.4585198700082895e-13, 4.716940764877479e-13,
                        ],
                        linf=[
                            7.774003663030271e-12, 9.183176441496244e-11,
                            4.5685344396417804e-11, 1.0534506600379245e-10,
                        ],
                        tspan=(0.0, 0.1),
                        atol=3.0e-13)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_wall_bc.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_wall_bc.jl"),
                        l2=[
                            0.040189107976346644,
                            0.04256154998030852,
                            0.03734120743842209,
                            0.10057425897733507,
                        ],
                        linf=[
                            0.24455374304626365,
                            0.2970686406973577,
                            0.29339040847600434,
                            0.5915610037764794,
                        ],
                        tspan=(0.0, 0.25),
                        surface_flux=FluxHLL(min_max_speed_naive))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_basic.jl" begin
    @test_trixi_include(default_example_unstructured(),
                        l2=[
                            0.0007213418215265047,
                            0.0006752337675043779,
                            0.0006437485997536973,
                            0.0014782883071363362,
                        ],
                        linf=[
                            0.004301288971032324,
                            0.005243995459478956,
                            0.004685630332338153,
                            0.01750217718347713,
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

@trixi_testset "elixir_euler_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_restart.jl"),
                        l2=[
                            0.0007213418215265047,
                            0.0006752337675043779,
                            0.0006437485997536973,
                            0.0014782883071363362,
                        ],
                        linf=[
                            0.004301288971032324,
                            0.005243995459478956,
                            0.004685630332338153,
                            0.01750217718347713,
                        ],
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

@trixi_testset "elixir_euler_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.06594600495903137,
                            0.10803914821786433,
                            0.10805946357846291,
                            0.1738171782368222,
                        ],
                        linf=[
                            0.31880214280781305,
                            0.3468488554333352,
                            0.34592958184413264,
                            0.784555926860546,
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

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        l2=[0.00018729339078205488],
                        linf=[0.0018997287705734278])
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
                            2.19945600e-01,
                            1.71050453e-01,
                            1.71050453e-01,
                            1.21719195e+00,
                        ],
                        linf=[
                            7.44218635e-01,
                            7.02887039e-01,
                            7.02887039e-01,
                            6.11732719e+00,
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

@trixi_testset "elixir_euler_time_series.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_time_series.jl"),
                        l2=[
                            6.984024099236519e-5,
                            6.289022520363763e-5,
                            6.550951878107466e-5,
                            0.00016222767700879948,
                        ],
                        linf=[
                            0.0005367823248620951,
                            0.000671293180158461,
                            0.0005656680962440319,
                            0.0013910024779804075,
                        ],
                        tspan=(0.0, 0.2),
                        # With the default `maxiters = 1` in coverage tests,
                        # there would be no time series to check against.
                        coverage_override=(maxiters = 20,))
    # Extra test that the `TimeSeries` callback creates reasonable data
    point_data_1 = time_series.affect!.point_data[1]
    @test all(isapprox.(point_data_1[1:4],
                        [1.9546882708551676, 1.9547149531788077,
                            1.9547142161310154, 3.821066781119142]))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_acoustics_gauss_wall.jl" begin
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
                        l2=[0.06418293357851637, 0.12085176618704108,
                            0.12085099342419513, 0.07743005602933221,
                            0.1622218916638482, 0.04044434425257972,
                            0.04044440614962498, 0.05735896706356321,
                            0.0020992340041681734],
                        linf=[0.1417000509328017, 0.3210578460652491, 0.335041095545175,
                            0.22500796423572675,
                            0.44230628074326406, 0.16743171716317784,
                            0.16745989278866702, 0.17700588224362557,
                            0.02692320090677309],
                        tspan=(0.0, 0.5))
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
                        l2=[5.377518922553881e-5, 0.09999999206243514,
                            0.09999999206243441, 0.1414213538550799,
                            8.770450430886394e-6, 0.0999999926130084,
                            0.0999999926130088, 0.14142135396487032,
                            1.1553833987291942e-5],
                        linf=[0.00039334982566352483, 0.14144904937275282,
                            0.14144904937277897, 0.20003315928443416,
                            6.826863293230012e-5, 0.14146512909995967,
                            0.14146512909994702, 0.20006706837452526,
                            0.00013645610312810813],
                        tspan=(0.0, 0.5))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_ec.jl"),
                        l2=[
                            0.6107326269462766,
                            0.48666631722018877,
                            0.48309775159067053,
                            0.29467422718511704,
                        ],
                        linf=[
                            2.776782342826098,
                            3.2158378644333707,
                            3.652920889487258,
                            2.052861364219655,
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

@trixi_testset "elixir_shallowwater_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            1.2164292510839076,
                            2.6118925543469468e-12,
                            2.459878823146057e-12,
                            1.2164292510839079,
                        ],
                        linf=[
                            1.5138512282315846,
                            4.706289937431355e-11,
                            4.913910192312011e-11,
                            1.513851228231574,
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

@trixi_testset "elixir_shallowwater_well_balanced.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            1.2164292510839085,
                            1.2643106818778908e-12,
                            1.269230436589819e-12,
                            1.2164292510839079,
                        ],
                        linf=[
                            1.513851228231562,
                            1.6670644673575802e-11,
                            1.8426585188623954e-11,
                            1.513851228231574,
                        ],
                        surface_flux=(FluxHydrostaticReconstruction(flux_lax_friedrichs,
                                                                    hydrostatic_reconstruction_audusse_etal),
                                      flux_nonconservative_audusse_etal),
                        tspan=(0.0, 0.2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_well_balanced.jl with flux_nonconservative_wintermeyer_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            1.2164292510839083,
                            2.590643638636187e-12,
                            2.388742604639019e-12,
                            1.2164292510839079,
                        ],
                        linf=[
                            1.5138512282315792,
                            4.761278694199934e-11,
                            4.910549479958249e-11,
                            1.513851228231574,
                        ],
                        surface_flux=(flux_wintermeyer_etal,
                                      flux_nonconservative_wintermeyer_etal),
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

@trixi_testset "elixir_shallowwater_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.001118134082248467,
                            0.044560486817464634,
                            0.01430926600634214,
                            5.089218476759981e-6,
                        ],
                        linf=[
                            0.007798727223654822,
                            0.34782952734839157,
                            0.11161614702628064,
                            2.6407324614341476e-5,
                        ],
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl with FluxHydrostaticReconstruction" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.0011196838135485918,
                            0.01542895635133927,
                            0.017082803023121197,
                            5.089218476759981e-6,
                        ],
                        linf=[
                            0.014299541415654371,
                            0.12783948113206955,
                            0.17626489583921323,
                            2.6407324614341476e-5,
                        ],
                        surface_flux=(FluxHydrostaticReconstruction(flux_hll,
                                                                    hydrostatic_reconstruction_audusse_etal),
                                      flux_nonconservative_audusse_etal),
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl with flux_nonconservative_wintermeyer_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.001118046975499805,
                            0.04455969246244461,
                            0.014298120235633432,
                            5.089218476759981e-6,
                        ],
                        linf=[
                            0.007776521213640031,
                            0.34768318303226353,
                            0.11075311228066198,
                            2.6407324614341476e-5,
                        ],
                        surface_flux=(flux_wintermeyer_etal,
                                      flux_nonconservative_wintermeyer_etal),
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_source_terms.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.0011196838135486059,
                            0.015428956351339451,
                            0.017082803023120943,
                            5.089218476759981e-6,
                        ],
                        linf=[
                            0.01429954141565526,
                            0.12783948113205668,
                            0.176264895839215,
                            2.6407324614341476e-5,
                        ],
                        surface_flux=(flux_hll,
                                      flux_nonconservative_fjordholm_etal),
                        tspan=(0.0, 0.025))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_dirichlet.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_dirichlet.jl"),
                        l2=[
                            1.1577518608938916e-5, 4.859252379740366e-13,
                            4.639600837197925e-13, 1.1577518608952174e-5,
                        ],
                        linf=[
                            8.3940638787805e-5, 1.1446362498574484e-10,
                            1.1124515748367981e-10, 8.39406387962427e-5,
                        ],
                        tspan=(0.0, 2.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_shallowwater_wall_bc_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_wall_bc_shockcapturing.jl"),
                        l2=[
                            0.0442113635677511, 0.1537465759364839, 0.16003586586203947,
                            6.225080477067782e-8,
                        ],
                        linf=[
                            0.6347820607387928, 2.0078125433846736, 2.530726684667019,
                            3.982097165344811e-7,
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

@trixi_testset "elixir_shallowwater_ec_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_ec_shockcapturing.jl"),
                        l2=[
                            0.612551520607341,
                            0.5039173660221961,
                            0.49136517934903523,
                            0.29467422718511704,
                        ],
                        linf=[
                            2.7636771472622197,
                            3.236168963021072,
                            3.3363936775653826,
                            2.052861364219655,
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

# TODO: FD; for now put the unstructured tests for the 2D FDSBP here.
@trixi_testset "FDSBP (central): elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "unstructured_2d_fdsbp"),
                                 "elixir_advection_basic.jl"),
                        l2=[0.0001105211407319266],
                        linf=[0.0004199363734466166])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "FDSBP (central): elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "unstructured_2d_fdsbp"),
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "FDSBP (central): elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "unstructured_2d_fdsbp"),
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "FDSBP (upwind): elixir_euler_source_terms_upwind.jl" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "unstructured_2d_fdsbp"),
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
                        atol=1.0e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "FDSBP (upwind): elixir_euler_source_terms_upwind.jl with LF splitting" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "unstructured_2d_fdsbp"),
                                 "elixir_euler_source_terms_upwind.jl"),
                        l2=[3.8300267071890586e-5,
                            5.295846741663533e-5,
                            5.295846741663526e-5,
                            0.00017564759295593478],
                        linf=[0.00018810716496542312,
                            0.0003794187430412599,
                            0.0003794187430412599,
                            0.0009632958510650269],
                        tspan=(0.0, 0.025),
                        flux_splitting=splitting_lax_friedrichs,
                        atol=1.0e-10)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "FDSBP (upwind): elixir_euler_free_stream_upwind.jl" begin
    @test_trixi_include(joinpath(pkgdir(Trixi, "examples", "unstructured_2d_fdsbp"),
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
                        atol=1.0e-10)
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

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
