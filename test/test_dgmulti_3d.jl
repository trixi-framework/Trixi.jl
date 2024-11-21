module TestExamplesDGMulti3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "dgmulti_3d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "DGMulti 3D" begin
#! format: noindent

# 3d tet/hex tests
@trixi_testset "elixir_euler_weakform.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        l2=[
                            0.000354593110864001, 0.00041301573702385284,
                            0.00037934556184883277, 0.0003525767114354012,
                            0.0013917457634530887
                        ],
                        linf=[
                            0.0036608123230692513, 0.005625540942772123,
                            0.0030565781898950206, 0.004158099048202857,
                            0.01932716837214299
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

@trixi_testset "elixir_euler_weakform.jl (EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        surface_integral=SurfaceIntegralWeakForm(flux_ranocha),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha),
                        # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.014932088450136542,
                            0.017080219613061526,
                            0.016589517840793006,
                            0.015905000907070196,
                            0.03903416208587798
                        ] ./ sqrt(8),
                        linf=[
                            0.06856547797256729,
                            0.08225664880340489,
                            0.06925055630951782,
                            0.06913016119820181,
                            0.19161418499621874
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

@trixi_testset "elixir_euler_weakform.jl (Hexahedral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        element_type=Hex(),
                        surface_integral=SurfaceIntegralWeakForm(FluxHLL(min_max_speed_naive)),
                        # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.00030580190715769566,
                            0.00040146357607439464,
                            0.00040146357607564597,
                            0.000401463576075708,
                            0.0015749412434154315
                        ] ./ sqrt(8),
                        linf=[
                            0.00036910287847780054,
                            0.00042659774184228283,
                            0.0004265977427213574,
                            0.00042659774250686233,
                            0.00143803344597071
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

@trixi_testset "elixir_euler_curved.jl (Hex elements, SBP, flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_curved.jl"),
                        l2=[
                            0.0019393929700612259,
                            0.003213659298633126,
                            0.003203104361527826,
                            0.0019407707245105426,
                            0.0109274471764788
                        ],
                        linf=[
                            0.01914151956454324,
                            0.0270195960766606,
                            0.026891238631389536,
                            0.019817504336972602,
                            0.09645660501766873
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

@trixi_testset "elixir_euler_curved.jl (Hex elements, GaussSBP, flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_curved.jl"),
                        approximation_type=GaussSBP(),
                        l2=[
                            0.0026311315195097097, 0.002914422404496567,
                            0.0029138891106640368, 0.002615140832315232,
                            0.006881528610616624
                        ],
                        linf=[
                            0.02099611487415931, 0.021314522450152307,
                            0.021288322783027613, 0.020273381695449455,
                            0.05259874039006007
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

@trixi_testset "elixir_euler_weakform_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
                        l2=[
                            0.00036475807571383924, 0.00043404536371780537,
                            0.0003985850214093045, 0.0003683451584072326,
                            0.00143503620472638
                        ],
                        linf=[
                            0.0032278615418719347, 0.005620238272054934,
                            0.0030514261010661237, 0.0039871165455998,
                            0.019282771780667396
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

@trixi_testset "elixir_euler_weakform_periodic.jl (Hexahedral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
                        element_type=Hex(),
                        surface_integral=SurfaceIntegralWeakForm(FluxHLL(min_max_speed_naive)),
                        # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.00034230612468547436,
                            0.00044397204714598747,
                            0.0004439720471461567,
                            0.0004439720471464591,
                            0.0016639410646990126
                        ] ./ sqrt(8),
                        linf=[
                            0.0003674374460325147,
                            0.0004253921341716982,
                            0.0004253921340786615,
                            0.0004253921340831024,
                            0.0014333414071048267
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

@trixi_testset "elixir_euler_weakform_periodic.jl (Hexahedral elements, SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
                        element_type=Hex(),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha),
                        surface_integral=SurfaceIntegralWeakForm(flux_ranocha),
                        approximation_type=SBP(),
                        # division by sqrt(8.0) corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.001712443468716032,
                            0.002491315550718859,
                            0.0024913155507195303,
                            0.002491315550720031,
                            0.008585818982343299
                        ] ./ sqrt(8),
                        linf=[
                            0.003810078279323559,
                            0.004998778644230928,
                            0.004998778643986235,
                            0.0049987786444081195,
                            0.016455044373650196
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

@trixi_testset "elixir_euler_taylor_green_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
                        polydeg=3, tspan=(0.0, 1.0), cells_per_dimension=(2, 2, 2),
                        l2=[
                            0.0003612827827560599,
                            0.06219350883951729,
                            0.062193508839503864,
                            0.08121963221634831,
                            0.07082703570808184
                        ],
                        linf=[
                            0.0007893509649821162,
                            0.1481953939988877,
                            0.14819539399791176,
                            0.14847291108358926,
                            0.21313533492212855
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

@trixi_testset "elixir_euler_taylor_green_vortex.jl (GaussSBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_taylor_green_vortex.jl"),
                        polydeg=3, approximation_type=GaussSBP(), tspan=(0.0, 1.0),
                        cells_per_dimension=(2, 2, 2),
                        l2=[
                            0.00036128278275524326,
                            0.062193508839511434,
                            0.06219350883949677,
                            0.08121963221635205,
                            0.07082703570765223
                        ],
                        linf=[
                            0.000789350964946367,
                            0.14819539399525805,
                            0.14819539399590542,
                            0.14847291107658706,
                            0.21313533492059378
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

@trixi_testset "elixir_euler_weakform_periodic.jl (FD SBP)" begin
    global D = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                                   derivative_order = 1,
                                   accuracy_order = 2,
                                   xmin = 0.0, xmax = 1.0,
                                   N = 8)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
                        element_type=Hex(),
                        cells_per_dimension=(2, 2, 2),
                        approximation_type=D,
                        l2=[
                            0.0024092707138829925,
                            0.003318758964118284,
                            0.0033187589641182386,
                            0.003318758964118252,
                            0.012689348410504253
                        ],
                        linf=[
                            0.006118565824207778,
                            0.008486456080185167,
                            0.008486456080180282,
                            0.008486456080185611,
                            0.035113544599208346
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

@trixi_testset "elixir_euler_weakform_periodic.jl (FD SBP, EC)" begin
    global D = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                                   derivative_order = 1,
                                   accuracy_order = 2,
                                   xmin = 0.0, xmax = 1.0,
                                   N = 8)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform_periodic.jl"),
                        element_type=Hex(),
                        cells_per_dimension=(2, 2, 2),
                        approximation_type=D,
                        volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha),
                        surface_integral=SurfaceIntegralWeakForm(flux_ranocha),
                        l2=[
                            0.0034543609010604407,
                            0.004944363692625066,
                            0.0049443636926250435,
                            0.004944363692625037,
                            0.01788695279620914
                        ],
                        linf=[
                            0.013861851418853988,
                            0.02126572106620328,
                            0.021265721066209053,
                            0.021265721066210386,
                            0.0771455289446683
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

@trixi_testset "elixir_euler_fdsbp_periodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_fdsbp_periodic.jl"),
                        l2=[
                            7.561896970325353e-5,
                            6.884047859361093e-5,
                            6.884047859363204e-5,
                            6.884047859361148e-5,
                            0.000201107274617457
                        ],
                        linf=[
                            0.0001337520020225913,
                            0.00011571467799287305,
                            0.0001157146779990903,
                            0.00011571467799376123,
                            0.0003446082308800058
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

@trixi_testset "elixir_advection_tensor_wedge.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_tensor_wedge.jl"),
                        l2=[2.30487910e-04],
                        linf=[6.31795281e-04])
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
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
