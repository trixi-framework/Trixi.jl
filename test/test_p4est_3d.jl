@testsnippet P4estMesh3D begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")
end

@testitem "P4estMesh3D: elixir_advection_basic.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_advection_limiter_liu_zhang.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(examples_dir(), "tree_3d_dgsem",
                                 "elixir_advection_limiter_liu_zhang.jl"),
                        mesh=P4estMesh((4, 4, 4), polydeg = 3,
                                       coordinates_min = (-1.0, -1.0, -1.0),
                                       coordinates_max = (1.0, 1.0, 1.0),
                                       initial_refinement_level = 1,
                                       periodicity = true),
                        l2=[0.3917747509079011],
                        linf=[1.3569114615249707])
    u = Trixi.wrap_array_native(sol.u[end], semi)
    # matches thresholds = (1e-3,) up to a tolerance
    @test minimum(u) > 1e-3 - 10 * eps()
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_advection_unstructured_curved.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_unstructured_curved.jl"),
                        l2=[0.0004750004258546538],
                        linf=[0.026527551737137167])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_advection_nonconforming.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
                        l2=[0.00253595715323843],
                        linf=[0.016486952252155795])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_advection_amr.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[9.773852895157622e-6],
                        linf=[0.0005853874124926162])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_advection_amr_unstructured_curved.jl" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_amr_unstructured_curved.jl"),
                        l2=[1.6163120948209677e-5],
                        linf=[0.0010572201890564834],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_advection_cubed_sphere.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_cubed_sphere.jl"),
                        l2=[0.002006918015656413],
                        linf=[0.027655117058380085])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_advection_restart.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[0.002590388934758452],
                        linf=[0.01840757696885409])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_source_terms_nonconforming_unstructured_curved.jl" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
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

    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_unstructured_curved.jl"),
                        surface_flux=FluxPlusDissipation(flux_ranocha,
                                                         DissipationMatrixWintersEtal()),
                        l2=[
                            4.068002997087932e-5,
                            4.4742882348806466e-5,
                            5.101817697733163e-5,
                            5.100410876233901e-5,
                            0.000199848133462063
                        ],
                        linf=[
                            0.0013080357114820806,
                            0.0028524316301083985,
                            0.0019100643150573582,
                            0.0024800222220195955,
                            0.00830424488849335
                        ],
                        tspan=(0.0, 0.01))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_source_terms_nonperiodic.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_free_stream.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_free_stream.jl (VolumeIntegralPureLGLFiniteVolume)" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        volume_integral=VolumeIntegralPureLGLFiniteVolume(flux_hllc),
                        cfl=1.0,
                        l2=[
                            1.9974155101075348e-15,
                            1.1961083306610996e-14,
                            1.2872974022570237e-14,
                            1.5462968285712546e-14,
                            2.6094715468434938e-14
                        ],
                        linf=[
                            2.828848266744899e-13,
                            1.6810441927361808e-12,
                            2.784633634789202e-12,
                            2.8425040099477883e-12,
                            3.7339020764193265e-12
                        ],
                        tspan=(0.0, 0.03))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_free_stream_extruded.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_free_stream_extruded_fvO2.jl" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_free_stream_extruded_fvO2.jl"),
                        l2=[
                            1.5286679406524262e-16,
                            4.950311927136664e-16,
                            4.69913693630953e-16,
                            4.012046698403924e-16,
                            2.0419659904946413e-15
                        ],
                        linf=[
                            2.220446049250313e-15,
                            3.3029134982598407e-15,
                            7.716050021144838e-15,
                            1.27675647831893e-14,
                            3.019806626980426e-14
                        ],
                        tspan=(0.0, 0.1))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_free_stream_boundaries.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_free_stream_boundaries_float32.jl" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
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
                        RealT_for_test_tolerances=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_free_stream_extruded.jl with HLLC FLux" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_ec.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
                        tspan=(0.0, 0.2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_ec.jl (flux_chandrashekar)" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
                        volume_flux=flux_chandrashekar)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_sedov.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_sedov_sc_subcell.jl (positivity bounds)" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_sc_subcell.jl"),
                        l2=[
                            0.19427117014566905,
                            0.07557679851688888,
                            0.07557679851688896,
                            0.07557679851688917,
                            0.3713893373335463
                        ],
                        linf=[
                            2.754292081408987,
                            1.888627709320322,
                            1.8886277093203232,
                            1.888627709320322,
                            4.971280431903264
                        ],
                        tspan=(0.0, 0.3))
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15_000)
end

@testitem "P4estMesh3D: elixir_euler_sedov_sc_subcell.jl (local bounds)" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_sc_subcell.jl"),
                        local_twosided_variables_cons=["rho"],
                        local_onesided_variables_nonlinear=[(entropy_guermond_etal,
                                                             min)],
                        max_iterations_newton=30,
                        l2=[
                            0.16504564013491585,
                            0.06461384162458203,
                            0.06461384162461223,
                            0.06461384162461678,
                            0.36193245790622036
                        ],
                        linf=[
                            0.9138327077620716,
                            0.5707102472596818,
                            0.5707102472739252,
                            0.5707102472781822,
                            4.777595503303726
                        ],
                        tspan=(0.0, 0.3))
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15_000)
end

@testitem "P4estMesh3D: elixir_euler_sedov_sc_subcell.jl (local bounds, nonperiodic)" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_sc_subcell.jl"),
                        local_twosided_variables_cons=["rho"],
                        local_onesided_variables_nonlinear=[(entropy_guermond_etal,
                                                             min)],
                        max_iterations_newton=50,
                        periodicity=false,
                        boundary_conditions=BoundaryConditionDirichlet(initial_condition),
                        l2=[
                            0.16504564013491585,
                            0.06461384162458203,
                            0.06461384162461223,
                            0.06461384162461678,
                            0.36193245790622036
                        ],
                        linf=[
                            0.9138327077620716,
                            0.5707102472596818,
                            0.5707102472739252,
                            0.5707102472781822,
                            4.777595503303726
                        ],
                        tspan=(0.0, 0.3))
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15_000)
end

@testitem "P4estMesh3D: elixir_euler_sedov.jl (HLLE)" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_source_terms_nonconforming_earth.jl" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonconforming_earth.jl"),
                        l2=[
                            2.6344746702983662e-6,
                            2.6658075090562718e-6,
                            2.704786453209594e-6,
                            2.703043852342655e-6,
                            1.0642547599835283e-5
                        ],
                        linf=[
                            1.8922755265826652e-5,
                            2.5970792253637853e-5,
                            2.59662162456209e-5,
                            2.596621623851547e-5,
                            6.590474876944086e-5
                        ],
                        # Decrease tolerance of adaptive time stepping to get similar results across different systems
                        abstol=1.0e-11, reltol=1.0e-11,
                        tspan=(0.0, 2.0e4))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_circular_wind_nonconforming.jl" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
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
                        tspan=(0.0, 2e2))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_baroclinic_instability.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
                        abstol=1.0e-9, reltol=1.0e-9)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_source_terms_nonperiodic_hohqmesh.jl" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_tandem_spheres.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_tandem_spheres.jl"),
                        # Strict tols to avoid issues with different systems (CI vs local machines)
                        tspan=(0.0, 1e-3), tols=1e-8,
                        l2=[
                            1.4563595831943405e-6,
                            1.1433846164578946e-6,
                            6.45964041497078e-7,
                            6.446228703680879e-7,
                            3.648646693902689e-6
                        ],
                        linf=[
                            0.08928818652777637,
                            0.09007801634411498,
                            0.04241320785718675,
                            0.04253018852293939,
                            0.22552627764127076
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_free_stream_hybrid_mesh.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_free_stream_hybrid_mesh.jl"),
                        l2=[
                            6.459343647140878e-16,
                            1.6402804603545678e-15,
                            2.0956682982886904e-15,
                            2.3267454534698676e-15,
                            3.802615222445529e-15
                        ],
                        linf=[
                            1.1657341758564144e-14,
                            1.6056600493641326e-14,
                            3.5665914666083154e-14,
                            3.68594044175552e-14,
                            6.217248937900877e-14
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_mhd_alfven_wave_er.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_er.jl"),
                        l2=[
                            0.005134130040244181,
                            0.010230227879059531,
                            0.006830967920339279,
                            0.013009869849363363,
                            0.014156421580732355,
                            0.010057716622589004,
                            0.0065840204031814625,
                            0.012477696035301746,
                            0.001199375886326767
                        ],
                        linf=[
                            0.02034957886348243,
                            0.04516283172637105,
                            0.024173565140199412,
                            0.050144513086875855,
                            0.06545980419975983,
                            0.0433659188690384,
                            0.038041617206837075,
                            0.051518104947560955,
                            0.00904840866390417
                        ])
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15_000)
end

@testitem "P4estMesh3D: elixir_mhd_alfven_wave_nonconforming.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.00017971806092102337,
                            0.0006136967511485613,
                            0.00022822690442389585,
                            0.0007226600407757523,
                            0.0007161606001252864,
                            0.0006066793297079816,
                            0.00027683929523419356,
                            0.0007102147790412794,
                            0.00018359046756855172
                        ],
                        linf=[
                            0.001149714620982456,
                            0.004384918175901309,
                            0.0012563090800819937,
                            0.0046327018891871435,
                            0.003125895329334938,
                            0.004268429815377117,
                            0.001771589463308465,
                            0.005043147336750528,
                            0.0014542473645139543
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_mhd_alfven_wave_nonconforming.jl (combined fluxes)" setup=[
    Setup,
    P4estMesh3D
] tags=[:p4est_part2] begin
    # Load the fluxes combining conservative and nonconservative terms,
    # which are equivalent to the fluxes used in the elixir below.
    # We do not need to run this simulation, so we replace `sol = solve(...)`
    # by `sol = nothing`.
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR,
                           "elixir_mhd_alfven_wave_combined_fluxes_nonperiodic.jl"),
                  sol = nothing)

    # Use the same reference values as for the separate fluxes
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonconforming.jl"),
                        l2=[
                            0.00017971806092102337,
                            0.0006136967511485613,
                            0.00022822690442389585,
                            0.0007226600407757523,
                            0.0007161606001252864,
                            0.0006066793297079816,
                            0.00027683929523419356,
                            0.0007102147790412794,
                            0.00018359046756855172
                        ],
                        linf=[
                            0.001149714620982456,
                            0.004384918175901309,
                            0.0012563090800819937,
                            0.0046327018891871435,
                            0.003125895329334938,
                            0.004268429815377117,
                            0.001771589463308465,
                            0.005043147336750528,
                            0.0014542473645139543
                        ],
                        solver=DGSEM(polydeg = 3,
                                     surface_flux = flux_hlle_nonconservative_powell,
                                     volume_integral = VolumeIntegralFluxDifferencing(flux_hindenlang_gassner_nonconservative_powell)),
                        tspan=(0.0, 0.25))

    # Make sure that the combined fluxes are used at mortars
    @test Trixi.nmortars(semi.solver, semi.cache) > 0

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_mhd_alfven_wave_nonperiodic.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_mhd_alfven_wave_nonperiodic.jl"),
                        l2=[
                            0.0001795929488059728,
                            0.0006165064652297039,
                            0.00022508366253058242,
                            0.0007252370187585684,
                            0.0007091879657138232,
                            0.0006104452763743072,
                            0.00027600132305446366,
                            0.0007168330368898018,
                            0.00020630565764640023
                        ],
                        linf=[
                            0.0011186617893810258,
                            0.004669712257139982,
                            0.0017793465002012357,
                            0.0047545329934178815,
                            0.004206493885438256,
                            0.004508985775530383,
                            0.001311986834732104,
                            0.005049041672472443,
                            0.0010719469892041644
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_mhd_shockcapturing_amr.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shockcapturing_amr.jl"),
                        l2=[
                            0.003230574555611937,
                            0.0035055385332353607,
                            0.00367603144467607,
                            0.0035613609448183568,
                            0.011371218399250963,
                            0.003053930455986979,
                            0.004239667341302796,
                            0.0030571586497968673,
                            0.0001622024325472736
                        ],
                        linf=[
                            0.11596394565891233,
                            0.11916244400971865,
                            0.16735969384585622,
                            0.14459732569870357,
                            0.4673566414490513,
                            0.11018098674910393,
                            0.1618795525489103,
                            0.1099319721127251,
                            0.02318074464594008
                        ],
                        tspan=(0.0, 0.02))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_mhd_shockcapturing_subcell.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shockcapturing_subcell.jl"),
                        l2=[
                            0.006730143381488665,
                            0.008638200362125233,
                            0.008978149441900818,
                            0.008546444369127841,
                            0.028552826645613108,
                            0.005790728186386162,
                            0.007478371113001749,
                            0.0058453450179431,
                            0.0002768640406343677
                        ],
                        linf=[
                            0.31496137264807067,
                            0.27624270565879083,
                            0.5102616117027343,
                            0.28998837906363095,
                            0.9543964272940535,
                            0.2621044614125685,
                            0.22947789123430673,
                            0.2855579198808329,
                            0.03258652743969817
                        ],
                        tspan=(0.0, 0.04))
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15_000)
end

@testitem "P4estMesh3D: elixir_mhd_amr_entropy_bounded.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_amr_entropy_bounded.jl"),
                        l2=[
                            0.005429460998014611,
                            0.006185025311927924,
                            0.012171667165675916,
                            0.006181452482844896,
                            0.035065636950935425,
                            0.004954183875450467,
                            0.006581212549975798,
                            0.005008804366387311,
                            0.00046961153322887554
                        ],
                        linf=[
                            0.18655823822323492,
                            0.2039706973119556,
                            0.3701182388876679,
                            0.23329247633917313,
                            1.0345872724689724,
                            0.18407131753884975,
                            0.20599765274654913,
                            0.188852408277848,
                            0.04080792401038258
                        ],
                        tspan=(0.0, 0.04))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_linearizedeuler_convergence.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "P4estMesh3D: elixir_euler_weak_blast_wave_amr.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weak_blast_wave_amr.jl"),
                        l2=[
                            0.006397126301849624,
                            0.012720295217888025,
                            0.01318988235587869,
                            0.013097523277007118,
                            0.10426475175103048
                        ],
                        linf=[
                            0.3065455410983535,
                            0.6592659455041433,
                            0.6262896448939657,
                            0.5631502851030113,
                            3.0893593575020954
                        ],
                        tspan=(0.0, 0.0125))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
    # Check for conservation
    state_integrals = Trixi.integrate(sol.u[2], semi)
    initial_state_integrals = analysis_callback.affect!.initial_state_integrals

    @test isapprox(state_integrals[1], initial_state_integrals[1], atol = 1e-13)
    @test isapprox(state_integrals[2], initial_state_integrals[2], atol = 1e-13)
    @test isapprox(state_integrals[3], initial_state_integrals[3], atol = 1e-13)
    @test isapprox(state_integrals[4], initial_state_integrals[4], atol = 1e-13)
    @test isapprox(state_integrals[5], initial_state_integrals[5], atol = 1e-13)
end

# Surface integrals of the pressure computed via `AnalysisSurfaceIntegral` on a curved
# mesh of [-1, 1]^3. The pressure p = 2 + y (1 + x) (1 + z) yields
# ∮ p n_x dS = ∫ ∂p/∂x dV = 0, ∮ p n_y dS = ∫ ∂p/∂y dV = 8, and
# ∫_{y = 1} p dS = ∫∫ 2 + (1 + x) (1 + z) dx dz = 12.
# The mapping keeps the boundary planar but distorts the boundary faces.
@testsnippet SurfaceIntegralsCurvedMesh3D begin
    function surface_integrals_curved_mesh_3d()
        equations = CompressibleEulerEquations3D(1.4)
        function initial_condition(x, t, equations)
            p = 2 + x[2] * (1 + x[1]) * (1 + x[3])
            return prim2cons(SVector(1.0, 0.0, 0.0, 0.0, p), equations)
        end
        function mapping(xi, eta, zeta)
            return SVector(xi + 0.05 * sinpi(xi) * (2 + eta + zeta),
                           eta + 0.05 * sinpi(eta) * (2 + zeta + xi),
                           zeta + 0.05 * sinpi(zeta) * (2 + xi + eta))
        end
        polydeg = 3
        solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)
        mesh = P4estMesh((4, 4, 4); polydeg, mapping, periodicity = false)
        boundary_condition = BoundaryConditionDirichlet(initial_condition)
        boundary_conditions = (; x_neg = boundary_condition, x_pos = boundary_condition,
                               y_neg = boundary_condition, y_pos = boundary_condition,
                               z_neg = boundary_condition, z_pos = boundary_condition)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                            boundary_conditions)

        # With these free-stream values, the coefficients are the surface integrals
        # of p (n ⋅ ψ)
        aoa, rho_inf, u_inf, a_inf = 0.0, 1.0, 1.0, 2.0
        all_boundaries = (:x_neg, :x_pos, :y_neg, :y_pos, :z_neg, :z_pos)
        lift = AnalysisSurfaceIntegral(all_boundaries,
                                       LiftCoefficientPressure3D(aoa, rho_inf, u_inf,
                                                                 a_inf))
        drag = AnalysisSurfaceIntegral(all_boundaries,
                                       DragCoefficientPressure3D(aoa, rho_inf, u_inf,
                                                                 a_inf))
        lift_top = AnalysisSurfaceIntegral((:y_pos,),
                                           LiftCoefficientPressure3D(aoa, rho_inf, u_inf,
                                                                     a_inf))

        # The right-hand side evaluation prolongs the solution to the boundaries
        u_ode = compute_coefficients(0.0, semi)
        du_ode = similar(u_ode)
        Trixi.rhs_hyperbolic!(du_ode, u_ode, semi, 0.0)
        # `wrap_array` may not keep `u_ode` and `du_ode` alive
        GC.@preserve u_ode du_ode begin
            u = Trixi.wrap_array(u_ode, semi)
            du = Trixi.wrap_array(du_ode, semi)
            return (Trixi.analyze(lift, du, u, 0.0, semi),
                    Trixi.analyze(drag, du, u, 0.0, semi),
                    Trixi.analyze(lift_top, du, u, 0.0, semi))
        end
    end
end

@testitem "P4estMesh3D: AnalysisSurfaceIntegral on curved mesh" setup=[
    Setup,
    SurfaceIntegralsCurvedMesh3D
] tags=[:p4est_part2] begin
    lift, drag, lift_top = surface_integrals_curved_mesh_3d()
    @test isapprox(lift, 8; atol = 1.0e-5)
    @test isapprox(drag, 0; atol = 1.0e-5)
    @test isapprox(lift_top, 12; atol = 1.0e-5)
end

@testitem "P4estMesh3D: elixir_euler_ONERA_M6_wing.jl" setup=[Setup, P4estMesh3D] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_ONERA_M6_wing.jl"),
                        l2=[
                            5.327066020924117e-8,
                            2.80976570476027e-8,
                            4.3866299071977515e-8,
                            2.7370729466369376e-8,
                            1.5199788880307453e-7
                        ],
                        linf=[
                            0.05744067831996991,
                            0.04819836936434463,
                            0.05781352954447377,
                            0.005198158826601687,
                            0.1647489333623069
                        ],
                        tspan=(0.0, 2e-8))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

# Multi-ion MHD tests
