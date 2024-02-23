module TestExamplesUnstructuredMesh2D

# TODO: TrixiShallowWater: move any wet/dry and two layer tests

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
                            0.00010978828464875207,
                            0.00013010359527356914,
                            0.00013010359527326057,
                            0.0002987656724828824,
                        ],
                        linf=[
                            0.00638626102818618,
                            0.009804042508242183,
                            0.009804042508253286,
                            0.02183139311614468,
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
                            3.3937971107485363e-14,
                            2.447586447887882e-13,
                            1.4585205789296455e-13,
                            4.716993468962946e-13,
                        ],
                        linf=[
                            8.804734719092266e-12,
                            6.261270668606045e-11,
                            2.93670088247211e-11,
                            1.205400224080222e-10,
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
                            0.6106939484178353,
                            0.48586236867426724,
                            0.48234490854514356,
                            0.29467422718511727,
                        ],
                        linf=[
                            2.775979948281604,
                            3.1721242154451548,
                            3.5713448319601393,
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
                            1.1636046671473883e-12,
                            1.2164292510839079,
                        ],
                        linf=[
                            1.5138512282315846,
                            4.998482888288039e-11,
                            2.0246214978154587e-11,
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
                            7.46884905098358e-13,
                            1.2164292510839079,
                        ],
                        linf=[
                            1.513851228231562,
                            1.6287765844373185e-11,
                            6.8766999132716964e-12,
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

@trixi_testset "elixir_shallowwater_well_balanced.jl with flux_nonconservative_ersing_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_well_balanced.jl"),
                        l2=[
                            1.2164292510839083,
                            2.590643638636187e-12,
                            1.0945471514840143e-12,
                            1.2164292510839079,
                        ],
                        linf=[
                            1.5138512282315792,
                            5.0276441977281156e-11,
                            1.9816934589292803e-11,
                            1.513851228231574,
                        ],
                        surface_flux=(flux_wintermeyer_etal,
                                      flux_nonconservative_ersing_etal),
                        volume_flux=(flux_wintermeyer_etal,
                                     flux_nonconservative_ersing_etal),
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
                            0.0011197623982310795,
                            0.04456344888447023,
                            0.014317376629669337,
                            5.089218476758975e-6,
                        ],
                        linf=[
                            0.007835284004819698,
                            0.3486891284278597,
                            0.11242778979399048,
                            2.6407324614119432e-5,
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
                            0.0011197139793938152,
                            0.015430259691310781,
                            0.017081031802719724,
                            5.089218476758271e-6,
                        ],
                        linf=[
                            0.014300809338967824,
                            0.12783372461225184,
                            0.17625472321992852,
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

@trixi_testset "elixir_shallowwater_source_terms.jl with flux_nonconservative_ersing_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        l2=[
                            0.0011196687776346434,
                            0.044562672453443995,
                            0.014306265289763618,
                            5.089218476759981e-6,
                        ],
                        linf=[
                            0.007825021762002393,
                            0.348550815397918,
                            0.1115517935018282,
                            2.6407324614341476e-5,
                        ],
                        surface_flux=(flux_wintermeyer_etal,
                                      flux_nonconservative_ersing_etal),
                        volume_flux=(flux_wintermeyer_etal,
                                     flux_nonconservative_ersing_etal),
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
                            0.0011197139793938727,
                            0.015430259691311309,
                            0.017081031802719554,
                            5.089218476759981e-6,
                        ],
                        linf=[
                            0.014300809338967824,
                            0.12783372461224918,
                            0.17625472321993918,
                            2.6407324614341476e-5,
                        ],
                        surface_flux=(flux_hll, flux_nonconservative_fjordholm_etal),
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
                            1.1577518608940115e-5,
                            4.867189932537344e-13,
                            4.647273240470541e-13,
                            1.1577518608933468e-5,
                        ],
                        linf=[
                            8.394063878602864e-5,
                            1.1469760027632646e-10,
                            1.1146619484429974e-10,
                            8.394063879602065e-5,
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
                            0.04444388691670699,
                            0.1527771788033111,
                            0.1593763537203512,
                            6.225080476986749e-8,
                        ],
                        linf=[
                            0.6526506870169639,
                            1.980765893182952,
                            2.4807635459119757,
                            3.982097158683473e-7,
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
                            0.6124656312639043,
                            0.504371951785709,
                            0.49180896200746366,
                            0.29467422718511727,
                        ],
                        linf=[
                            2.7639232436274392,
                            3.3985508653311767,
                            3.3330308209196224,
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

@trixi_testset "elixir_shallowwater_three_mound_dam_break.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_three_mound_dam_break.jl"),
                        l2=[
                            0.0892957892027502,
                            0.30648836484407915,
                            2.28712547616214e-15,
                            0.0008778654298684622,
                        ],
                        linf=[
                            0.850329472915091,
                            2.330631694956507,
                            5.783660020252348e-14,
                            0.04326237921249021,
                        ],
                        basis=LobattoLegendreBasis(3),
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

@trixi_testset "elixir_shallowwater_twolayer_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_twolayer_convergence.jl"),
                        l2=[0.0007935561625451243, 0.008825315509943844,
                            0.002429969315645897,
                            0.0007580145888686304, 0.004495741879625235,
                            0.0015758146898767814,
                            6.849532064729749e-6],
                        linf=[0.0059205195991136605, 0.08072126590166251,
                            0.03463806075399023,
                            0.005884818649227186, 0.042658506561995546,
                            0.014125956138838602, 2.5829318284764646e-5],
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

@trixi_testset "elixir_shallowwater_twolayer_well_balanced.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_twolayer_well_balanced.jl"),
                        l2=[4.706532184998499e-16, 1.1215950712872183e-15,
                            6.7822712922421565e-16,
                            0.002192812926266047, 5.506855295923691e-15,
                            3.3105180099689275e-15,
                            0.0021928129262660085],
                        linf=[4.468647674116255e-15, 1.3607872120431166e-14,
                            9.557155049520056e-15,
                            0.024280130945632084, 6.68910907640583e-14,
                            4.7000983997100496e-14,
                            0.024280130945632732],
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

@trixi_testset "elixir_shallowwater_twolayer_dam_break.jl with flux_lax_friedrichs" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_shallowwater_twolayer_dam_break.jl"),
                        l2=[0.012447632879122346, 0.012361250464676683,
                            0.0009551519536340908,
                            0.09119400061322577, 0.015276216721920347,
                            0.0012126995108983853, 0.09991983966647647],
                        linf=[0.044305765721807444, 0.03279620980615845,
                            0.010754320388190101,
                            0.111309922939555, 0.03663360204931427,
                            0.014332822306649284,
                            0.10000000000000003],
                        surface_flux=(flux_lax_friedrichs,
                                      flux_nonconservative_ersing_etal),
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
                        linf=[2.353373051988683e-10,
                            2.801543719233024e-11,
                            3.930469838486772e-11,
                            4.61017890529547e-11],
                        tspan=(0.0, 0.1),
                        atol=1.0e-11)
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
