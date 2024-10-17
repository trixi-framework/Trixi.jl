module TestExamplesP4estMesh3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "P4estMesh3D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_unstructured_curved.jl"),
                        l2=[0.0004750004258546538],
                        linf=[0.026527551737137167])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
                        l2=[0.00253595715323843],
                        linf=[0.016486952252155795])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[9.773852895157622e-6],
                        linf=[0.0005853874124926162],
                        coverage_override=(maxiters = 6, initial_refinement_level = 1,
                                           base_level = 1, med_level = 2, max_level = 3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_amr_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_unstructured_curved.jl"),
                        l2=[1.6163120948209677e-5],
                        linf=[0.0010572201890564834],
                        tspan=(0.0, 1.0),
                        coverage_override=(maxiters = 6, initial_refinement_level = 0,
                                           base_level = 0, med_level = 1, max_level = 2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_cubed_sphere.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_cubed_sphere.jl"),
                        l2=[0.002006918015656413],
                        linf=[0.027655117058380085])
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
                        l2=[0.002590388934758452],
                        linf=[0.01840757696885409],
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

@trixi_testset "elixir_euler_source_terms_nonconforming_unstructured_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_curved.jl"),
                        l2=[
                            4.070355207909268e-5,
                            4.4993257426833716e-5,
                            5.10588457841744e-5,
                            5.102840924036687e-5,
                            0.00019986264001630542
                        ],
                        linf=[
                            0.0016987332417202072,
                            0.003622956808262634,
                            0.002029576258317789,
                            0.0024206977281964193,
                            0.008526972236273522
                        ],
                        tspan=(0.0, 0.01))
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
                            0.0015106060984283647,
                            0.0014733349038567685,
                            0.00147333490385685,
                            0.001473334903856929,
                            0.0028149479453087093
                        ],
                        linf=[
                            0.008070806335238156,
                            0.009007245083113125,
                            0.009007245083121784,
                            0.009007245083102688,
                            0.01562861968368434
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

@trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        l2=[
                            5.162664597942288e-15,
                            1.941857343642486e-14,
                            2.0232366394187278e-14,
                            2.3381518645408552e-14,
                            7.083114561232324e-14
                        ],
                        linf=[
                            7.269740365245525e-13,
                            3.289868377720495e-12,
                            4.440087186807773e-12,
                            3.8686831516088205e-12,
                            9.412914891981927e-12
                        ],
                        tspan=(0.0, 0.03))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_free_stream_extruded.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_extruded.jl"),
                        l2=[
                            8.444868392439035e-16,
                            4.889826056731442e-15,
                            2.2921260987087585e-15,
                            4.268460455702414e-15,
                            1.1356712092620279e-14
                        ],
                        linf=[
                            7.749356711883593e-14,
                            2.8792246364872653e-13,
                            1.1121659149182506e-13,
                            3.3228975127030935e-13,
                            9.592326932761353e-13
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

@trixi_testset "elixir_euler_free_stream_boundaries.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_free_stream_boundaries.jl"),
                        l2=[
                            6.530157034651212e-16, 1.6057829680004379e-15,
                            3.31107455378537e-15, 3.908829498281281e-15,
                            5.048390610424672e-15
                        ],
                        linf=[
                            4.884981308350689e-15, 1.1921019726912618e-14,
                            1.5432100042289676e-14, 2.298161660974074e-14,
                            6.039613253960852e-14
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

@trixi_testset "elixir_euler_free_stream_boundaries_float32.jl" begin
    # Expected errors are taken from elixir_euler_free_stream_boundaries.jl
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_free_stream_boundaries_float32.jl"),
                        l2=[
                            Float32(6.530157034651212e-16),
                            Float32(1.6057829680004379e-15),
                            Float32(3.31107455378537e-15),
                            Float32(3.908829498281281e-15),
                            Float32(5.048390610424672e-15)
                        ],
                        linf=[
                            Float32(4.884981308350689e-15),
                            Float32(1.1921019726912618e-14),
                            Float32(1.5432100042289676e-14),
                            Float32(2.298161660974074e-14),
                            Float32(6.039613253960852e-14)
                        ],
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

@trixi_testset "elixir_euler_free_stream_extruded.jl with HLLC FLux" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream_extruded.jl"),
                        l2=[
                            8.444868392439035e-16,
                            4.889826056731442e-15,
                            2.2921260987087585e-15,
                            4.268460455702414e-15,
                            1.1356712092620279e-14
                        ],
                        linf=[
                            7.749356711883593e-14,
                            4.513472928735496e-13,
                            2.9790059308254513e-13,
                            1.057154364048074e-12,
                            1.6271428648906294e-12
                        ],
                        tspan=(0.0, 0.1),
                        surface_flux=flux_hllc)
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
                            0.010380390326164493,
                            0.006192950051354618,
                            0.005970674274073704,
                            0.005965831290564327,
                            0.02628875593094754
                        ],
                        linf=[
                            0.3326911600075694,
                            0.2824952141320467,
                            0.41401037398065543,
                            0.45574161423218573,
                            0.8099577682187109
                        ],
                        tspan=(0.0, 0.2),
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

@trixi_testset "elixir_euler_ec.jl (flux_chandrashekar)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.010368548525287055,
                            0.006216054794583285,
                            0.006020401857347216,
                            0.006019175682769779,
                            0.026228080232814154
                        ],
                        linf=[
                            0.3169376449662026,
                            0.28950510175646726,
                            0.4402523227566396,
                            0.4869168122387365,
                            0.7999141641954051
                        ],
                        tspan=(0.0, 0.2),
                        volume_flux=flux_chandrashekar,
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

@trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            7.82070951e-02,
                            4.33260474e-02,
                            4.33260474e-02,
                            4.33260474e-02,
                            3.75260911e-01
                        ],
                        linf=[
                            7.45329845e-01,
                            3.21754792e-01,
                            3.21754792e-01,
                            3.21754792e-01,
                            4.76151527e+00
                        ],
                        tspan=(0.0, 0.3),
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

@trixi_testset "elixir_euler_sedov.jl (HLLE)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            0.09946224487902565,
                            0.04863386374672001,
                            0.048633863746720116,
                            0.04863386374672032,
                            0.3751015774232693
                        ],
                        linf=[
                            0.789241521871487,
                            0.42046970270100276,
                            0.42046970270100276,
                            0.4204697027010028,
                            4.730877375538398
                        ],
                        tspan=(0.0, 0.3),
                        surface_flux=flux_hlle)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_nonconforming_earth.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_earth.jl"),
                        l2=[
                            6.040180337738628e-6,
                            5.4254175153621895e-6,
                            5.677698851333843e-6,
                            5.8017136892469794e-6,
                            1.3637854615117974e-5
                        ],
                        linf=[
                            0.00013996924184311865,
                            0.00013681539559939893,
                            0.00013681539539733834,
                            0.00013681539541021692,
                            0.00016833038543762058
                        ],
                        # Decrease tolerance of adaptive time stepping to get similar results across different systems
                        abstol=1.0e-11, reltol=1.0e-11,
                        coverage_override=(trees_per_cube_face = (1, 1), polydeg = 3)) # Prevent long compile time in CI
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_circular_wind_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_circular_wind_nonconforming.jl"),
                        l2=[
                            1.5737711609657832e-7,
                            3.8630261900166194e-5,
                            3.8672287531936816e-5,
                            3.6865116098660796e-5,
                            0.05508620970403884
                        ],
                        linf=[
                            2.268845333053271e-6,
                            0.000531462302113539,
                            0.0005314624461298934,
                            0.0005129931254772464,
                            0.7942778058932163
                        ],
                        tspan=(0.0, 2e2),
                        coverage_override=(trees_per_cube_face = (1, 1), polydeg = 3)) # Prevent long compile time in CI
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_baroclinic_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_baroclinic_instability.jl"),
                        l2=[
                            6.725093801700048e-7,
                            0.00021710076010951073,
                            0.0004386796338203878,
                            0.00020836270267103122,
                            0.07601887903440395
                        ],
                        linf=[
                            1.9107530539574924e-5,
                            0.02980358831035801,
                            0.048476331898047564,
                            0.02200137344113612,
                            4.848310144356219
                        ],
                        tspan=(0.0, 1e2),
                        # Decrease tolerance of adaptive time stepping to get similar results across different systems
                        abstol=1.0e-9, reltol=1.0e-9,
                        coverage_override=(trees_per_cube_face = (1, 1), polydeg = 3)) # Prevent long compile time in CI
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_nonperiodic_hohqmesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic_hohqmesh.jl"),
                        l2=[
                            0.0042023406458005464,
                            0.004122532789279737,
                            0.0042448149597303616,
                            0.0036361316700401765,
                            0.007389845952982495
                        ],
                        linf=[
                            0.04530610539892499,
                            0.02765695110527666,
                            0.05670295599308606,
                            0.048396544302230504,
                            0.1154589758186293
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

@trixi_testset "elixir_mhd_alfven_wave_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[0.00019018725889431733, 0.0006523517707148006,
                            0.0002401595437705759, 0.0007796920661427565,
                            0.0007095787460334334, 0.0006558819731628876,
                            0.0003565026134076906, 0.0007904654548841712,
                            9.437300326448332e-7],
                        linf=[0.0012482306861187897, 0.006408776208178299,
                            0.0016845452099629663, 0.0068711236542984555,
                            0.004626581522263695, 0.006614624811393632,
                            0.0030068344747734566, 0.008277825749754025,
                            1.3475027166309006e-5],
                        tspan=(0.0, 0.25),
                        coverage_override=(trees_per_dimension = (1, 1, 1),))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_shockcapturing_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shockcapturing_amr.jl"),
                        l2=[0.006297229188267704, 0.006436347763092648,
                            0.0071091348227321095, 0.00652953798427642,
                            0.0206148702828057, 0.005561406556411695,
                            0.007570747563696005, 0.005571060186513173,
                            3.888176398720913e-6],
                        linf=[0.20904050630623572, 0.1863002690612441,
                            0.2347653795205547, 0.19430178062881898,
                            0.6858488630270272, 0.15169972127018583,
                            0.22431157058134898, 0.16823638722404644,
                            0.0005208971463830214],
                        tspan=(0.0, 0.04),
                        coverage_override=(maxiters = 6, initial_refinement_level = 1,
                                           base_level = 1, max_level = 2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_amr_entropy_bounded.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_amr_entropy_bounded.jl"),
                        l2=[
                            0.005430176785094096,
                            0.006185803468926062,
                            0.012158513265762224,
                            0.006185144232789619,
                            0.03509140423905665,
                            0.004968215426326584,
                            0.006553519141867704,
                            0.005008885124643863,
                            5.165777182726578e-6
                        ],
                        linf=[
                            0.1864317840224794,
                            0.2041246899193812,
                            0.36992946717578445,
                            0.2327158690965257,
                            1.0368624176126007,
                            0.1846308291826353,
                            0.2062255411778191,
                            0.18955666546331185,
                            0.0005208969502913304
                        ],
                        tspan=(0.0, 0.04),
                        coverage_override=(maxiters = 6, initial_refinement_level = 1,
                                           base_level = 1, max_level = 2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_linearizedeuler_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_linearizedeuler_convergence.jl"),
                        l2=[
                            0.04452389418193219, 0.03688186699434862,
                            0.03688186699434861, 0.03688186699434858,
                            0.044523894181932186
                        ],
                        linf=[
                            0.2295447498696467, 0.058369658071546704,
                            0.05836965807154648, 0.05836965807154648, 0.2295447498696468
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

@trixi_testset "elixir_euler_weak_blast_wave_amr.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weak_blast_wave_amr.jl"),
                        l2=[
                            0.011345993108796831,
                            0.018525073963833696,
                            0.019102348105917946,
                            0.01920515438943838,
                            0.15060493968460148
                        ],
                        linf=[
                            0.2994949779783401,
                            0.5530175050084679,
                            0.5335803757792128,
                            0.5647252867336123,
                            3.6462732329242566
                        ],
                        tspan=(0.0, 0.025),
                        coverage_override=(maxiters = 6,))
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
    @test isapprox(state_integrals[5], initial_state_integrals[5], atol = 1e-13)
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end # module
