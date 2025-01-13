module TestExamplesDGMulti2D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = joinpath(examples_dir(), "dgmulti_2d")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "DGMulti 2D" begin
#! format: noindent

@trixi_testset "elixir_euler_weakform.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        cells_per_dimension=(4, 4),
                        surface_integral=SurfaceIntegralWeakForm(FluxHLL(min_max_speed_naive)),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.0013536930300254945,
                            0.0014315603442106193,
                            0.001431560344211359,
                            0.0047393341007602625
                        ] ./ 2.0,
                        linf=[
                            0.001514260921466004,
                            0.0020623991944839215,
                            0.002062399194485476,
                            0.004897700392503701
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

@trixi_testset "elixir_euler_weakform.jl (SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        cells_per_dimension=(4, 4),
                        approximation_type=SBP(),
                        surface_integral=SurfaceIntegralWeakForm(FluxHLL(min_max_speed_naive)),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.0074706882014934735,
                            0.005306220583603261,
                            0.005306220583613591,
                            0.014724842607716771
                        ] ./ 2.0,
                        linf=[
                            0.021563604940952885,
                            0.01359397832530762,
                            0.013593978324845324,
                            0.03270995869587523
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

@trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        cells_per_dimension=(4, 4),
                        element_type=Quad(),
                        surface_integral=SurfaceIntegralWeakForm(FluxHLL(min_max_speed_naive)),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.00031892254415307093,
                            0.00033637562986771894,
                            0.0003363756298680649,
                            0.0011100259064243145
                        ] ./ 2.0,
                        linf=[
                            0.001073298211445639,
                            0.0013568139808282087,
                            0.0013568139808290969,
                            0.0032249020004324613
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

@trixi_testset "elixir_euler_weakform.jl (EC) " begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        cells_per_dimension=(4, 4),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha),
                        surface_integral=SurfaceIntegralWeakForm(flux_ranocha),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.007801417730672109,
                            0.00708583561714128,
                            0.0070858356171393,
                            0.015217574294198809
                        ] ./ 2.0,
                        linf=[
                            0.011572828457858897,
                            0.013965298735070686,
                            0.01396529873508534,
                            0.04227683691807904
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

@trixi_testset "elixir_euler_weakform.jl (SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        cells_per_dimension=(4, 4),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha),
                        surface_integral=SurfaceIntegralWeakForm(flux_ranocha),
                        approximation_type=SBP(),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.01280067571168776,
                            0.010607599608273302,
                            0.010607599608239775,
                            0.026408338014056548
                        ] ./ 2.0,
                        linf=[
                            0.037983023185674814,
                            0.05321027922533417,
                            0.05321027922608157,
                            0.13392025411844033
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

@trixi_testset "elixir_euler_weakform.jl (Quadrilateral elements, SBP, EC)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        cells_per_dimension=(4, 4),
                        element_type=Quad(),
                        volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha),
                        surface_integral=SurfaceIntegralWeakForm(flux_ranocha),
                        approximation_type=SBP(),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.0029373718090697975,
                            0.0030629360605489465,
                            0.003062936060545615,
                            0.0068486089344859755
                        ] ./ 2.0,
                        linf=[
                            0.01360165305316885,
                            0.01267402847925303,
                            0.012674028479251254,
                            0.02210545278615017
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

@trixi_testset "elixir_euler_bilinear.jl (Bilinear quadrilateral elements, SBP, flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_bilinear.jl"),
                        l2=[
                            1.0259432774540821e-5, 9.014087689495575e-6,
                            9.01408768888544e-6, 2.738953324859446e-5
                        ],
                        linf=[
                            7.362605996297233e-5, 6.874189724781488e-5,
                            6.874189703509614e-5, 0.00019124355334110277
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

@trixi_testset "elixir_euler_curved.jl (Quadrilateral elements, SBP, flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_curved.jl"),
                        l2=[
                            1.7204593127904542e-5, 1.5921547179522804e-5,
                            1.5921547180107928e-5, 4.894071422525737e-5
                        ],
                        linf=[
                            0.00010525416937667842, 0.00010003778102718464,
                            0.00010003778071832059, 0.0003642628211952825
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

@trixi_testset "elixir_euler_curved.jl (Quadrilateral elements, GaussSBP, flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_curved.jl"),
                        approximation_type=GaussSBP(),
                        surface_integral=SurfaceIntegralWeakForm(FluxHLL(min_max_speed_naive)),
                        l2=[
                            3.4666312079259457e-6,
                            3.4392774480368986e-6,
                            3.439277447953705e-6,
                            1.0965598424665836e-5
                        ],
                        linf=[
                            1.1327280377004811e-5,
                            1.1343911926253725e-5,
                            1.1343911906935844e-5,
                            3.679582619220412e-5
                        ],
                        rtol=2 * sqrt(eps()))
    # Ensure that we do not have excessive memory allocations 
    # (e.g., from type instabilities) 
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_curved.jl (Triangular elements, Polynomial, weak formulation)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_curved.jl"),
                        element_type=Tri(), approximation_type=Polynomial(),
                        volume_integral=VolumeIntegralWeakForm(),
                        surface_integral=SurfaceIntegralWeakForm(FluxHLL(min_max_speed_naive)),
                        l2=[
                            7.905498158659466e-6,
                            8.731690809663625e-6,
                            8.731690811576996e-6,
                            2.9113296018693953e-5
                        ],
                        linf=[
                            3.298811230090237e-5,
                            4.032272476939269e-5,
                            4.032272526011127e-5,
                            0.00012013725458537294
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

@trixi_testset "elixir_euler_hohqmesh.jl (Quadrilateral elements, SBP, flux differencing)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_hohqmesh.jl"),
                        l2=[
                            0.0008153911341517156,
                            0.0007768159701964676,
                            0.00047902606811690694,
                            0.0015551846076348535
                        ],
                        linf=[
                            0.0029301131365355726,
                            0.0034427051471457304,
                            0.0028721569841545502,
                            0.011125365074589944
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

@trixi_testset "elixir_euler_weakform.jl (convergence)" begin
    mean_convergence = convergence_test(@__MODULE__,
                                        joinpath(EXAMPLES_DIR,
                                                 "elixir_euler_weakform.jl"), 2)
    @test isapprox(mean_convergence[:l2],
                   [
                       4.243843382379403,
                       4.128314378833922,
                       4.128314378397532,
                       4.081366752807379
                   ], rtol = 0.05)
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
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.0007492755162295128, 0.0007641875305302599,
                            0.0007641875305306243, 0.0024232389721009447
                        ],
                        linf=[
                            0.0015060064614331736, 0.0019371156800773726,
                            0.0019371156800769285, 0.004742431684202408
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

@trixi_testset "elixir_euler_triangulate_pkg_mesh.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_triangulate_pkg_mesh.jl"),
                        l2=[
                            2.344076909832665e-6, 1.8610002398709756e-6,
                            2.4095132179484066e-6, 6.37330249340445e-6
                        ],
                        linf=[
                            2.509979394305084e-5, 2.2683711321080935e-5,
                            2.6180377720841363e-5, 5.575278031910713e-5
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

@trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_kelvin_helmholtz_instability.jl"),
                        cells_per_dimension=(32, 32), tspan=(0.0, 0.2),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.11140378947116614,
                            0.06598161188703612,
                            0.10448953167839563,
                            0.16023209181809595
                        ] ./ 2.0,
                        linf=[
                            0.24033843177853664,
                            0.1659992245272325,
                            0.1235468309508845,
                            0.26911424973147735
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

@trixi_testset "elixir_euler_kelvin_helmholtz_instability.jl (Quadrilateral elements, GaussSBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_kelvin_helmholtz_instability.jl"),
                        cells_per_dimension=(32, 32), element_type=Quad(),
                        approximation_type=GaussSBP(), tspan=(0.0, 0.2),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.11141270656347146,
                            0.06598888014584121,
                            0.1044902203749932,
                            0.16023037364774995
                        ] ./ 2.0,
                        linf=[
                            0.2414760062126462,
                            0.1662111846065654,
                            0.12344140473946856,
                            0.26978428189564774
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

@trixi_testset "elixir_euler_rayleigh_taylor_instability.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_rayleigh_taylor_instability.jl"),
                        cells_per_dimension=(8, 8), tspan=(0.0, 0.2),
                        l2=[
                            0.07097806723891838, 0.005168550941966817,
                            0.013820912272220933, 0.03243357220022434
                        ],
                        linf=[
                            0.4783395896753895, 0.02244629340135818,
                            0.04023357731088538, 0.08515807256615027
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

@trixi_testset "elixir_euler_brown_minion_vortex.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_brown_minion_vortex.jl"),
                        cells_per_dimension=4, tspan=(0.0, 0.1),
                        l2=[
                            0.006680001611078062,
                            0.02151676347585447,
                            0.010696524235364626,
                            0.15052841129694647
                        ],
                        linf=[
                            0.01544756362800248,
                            0.09517304772476806,
                            0.021957154972646383,
                            0.33773439650806303
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

@trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
                        cells_per_dimension=4, tspan=(0.0, 0.1),
                        l2=[
                            0.05685148333985476,
                            0.04308122135907089,
                            0.043081221359070915,
                            0.21098131003847664
                        ],
                        linf=[
                            0.2360672306096051,
                            0.16684417686971842,
                            0.1668441768697189,
                            0.8572572782118661
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

@trixi_testset "elixir_euler_shockcapturing_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_curved.jl"),
                        cells_per_dimension=4, tspan=(0.0, 0.1),
                        l2=[
                            0.05565849298766252,
                            0.042322816017256494,
                            0.042322816017256466,
                            0.2064212098324083
                        ],
                        linf=[
                            0.23633287875008924,
                            0.16930148707515683,
                            0.16930148707515688,
                            0.8587706761131937
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

@trixi_testset "elixir_euler_weakform.jl (FD SBP)" begin
    global D = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                                   derivative_order = 1,
                                   accuracy_order = 4,
                                   xmin = 0.0, xmax = 1.0,
                                   N = 12)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        cells_per_dimension=(2, 2),
                        element_type=Quad(),
                        cfl=1.0,
                        approximation_type=D,
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.0008966318978421226,
                            0.0011418826379110242,
                            0.001141882637910878,
                            0.0030918374335671393
                        ] ./ 2.0,
                        linf=[
                            0.0015281525343109337,
                            0.00162430960401716,
                            0.0016243096040242655,
                            0.004447503691245913
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

@trixi_testset "elixir_euler_weakform.jl (FD SBP, EC)" begin
    global D = derivative_operator(SummationByPartsOperators.MattssonNordström2004(),
                                   derivative_order = 1,
                                   accuracy_order = 4,
                                   xmin = 0.0, xmax = 1.0,
                                   N = 12)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_weakform.jl"),
                        cells_per_dimension=(2, 2),
                        element_type=Quad(),
                        cfl=1.0,
                        approximation_type=D,
                        volume_integral=VolumeIntegralFluxDifferencing(flux_ranocha),
                        surface_integral=SurfaceIntegralWeakForm(flux_ranocha),
                        # division by 2.0 corresponds to normalization by the square root of the size of the domain
                        l2=[
                            0.0014018725496871129,
                            0.0015887007320868913,
                            0.001588700732086329,
                            0.003870926821031202
                        ] ./ 2.0,
                        linf=[
                            0.0029541996523780867,
                            0.0034520465226108854,
                            0.003452046522624652,
                            0.007677153211004928
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
                            1.333332033888785e-6, 2.044834627786368e-6,
                            2.0448346278315884e-6, 5.282189803437435e-6
                        ],
                        linf=[
                            2.7000151703315822e-6, 3.988595025372632e-6,
                            3.9885950240403645e-6, 8.848583036513702e-6
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

@trixi_testset "elixir_euler_fdsbp_periodic.jl (arbitrary reference domain)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_fdsbp_periodic.jl"),
                        xmin=-200.0, xmax=100.0, #= parameters for reference interval =#
                        surface_flux=FluxHLL(min_max_speed_naive),
                        l2=[
                            1.333332034149886e-6,
                            2.0448346280892024e-6,
                            2.0448346279766305e-6,
                            5.282189803510037e-6
                        ],
                        linf=[
                            2.700015170553627e-6,
                            3.988595024262409e-6,
                            3.988595024928543e-6,
                            8.84858303740188e-6
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

@trixi_testset "elixir_euler_fdsbp_periodic.jl (arbitrary reference and physical domains)" begin
    global D = periodic_derivative_operator(derivative_order = 1,
                                            accuracy_order = 4,
                                            xmin = -200.0,
                                            xmax = 100.0,
                                            N = 100)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_fdsbp_periodic.jl"),
                        approximation_type=D,
                        coordinates_min=(-3.0, -4.0), coordinates_max=(0.0, -1.0),
                        surface_flux=FluxHLL(min_max_speed_naive),
                        l2=[
                            0.07318831033918516,
                            0.10039910610067465,
                            0.1003991061006748,
                            0.2642450566234564
                        ],
                        linf=[
                            0.36081081739439735,
                            0.5244468027020845,
                            0.5244468027020814,
                            1.2210130256735705
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

@trixi_testset "elixir_euler_fdsbp_periodic.jl (CGSEM)" begin
    D_local = SummationByPartsOperators.legendre_derivative_operator(xmin = 0.0,
                                                                     xmax = 1.0,
                                                                     N = 4)
    mesh_local = SummationByPartsOperators.UniformPeriodicMesh1D(xmin = -1.0,
                                                                 xmax = 1.0,
                                                                 Nx = 10)
    global D = SummationByPartsOperators.couple_continuously(D_local, mesh_local)
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_fdsbp_periodic.jl"),
                        approximation_type=D,
                        surface_flux=FluxHLL(min_max_speed_naive),
                        l2=[
                            1.5440402410017893e-5,
                            1.4913189903083485e-5,
                            1.4913189902797073e-5,
                            2.6104615985156992e-5
                        ],
                        linf=[
                            4.16334345412217e-5,
                            5.067812788173143e-5,
                            5.067812786885284e-5,
                            9.887976803746312e-5
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

@trixi_testset "elixir_mhd_weak_blast_wave.jl (Quad)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
                        cells_per_dimension=4,
                        l2=[0.03906769915509508, 0.04923079758984701,
                            0.049230797589847136, 0.02660348840973283,
                            0.18054907061740028, 0.019195256934309846,
                            0.019195256934310016, 0.027856113419468087,
                            0.0016567799774264065],
                        linf=[0.16447597822733662, 0.244157345789029,
                            0.24415734578903472, 0.11982440036793476,
                            0.7450328339751362, 0.06357382685763713, 0.0635738268576378,
                            0.1058830287485999,
                            0.005740591170062146])
    # Ensure that we do not have excessive memory allocations 
    # (e.g., from type instabilities) 
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_weak_blast_wave.jl (Tri)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_weak_blast_wave.jl"),
                        cells_per_dimension=4, element_type=Tri(),
                        l2=[0.03372468091254386, 0.03971626483409167,
                            0.03971626483409208, 0.021427571421535722,
                            0.15079331840847413, 0.015716300366650286,
                            0.015716300366652128, 0.022365252076279075,
                            0.0009232971979900358],
                        linf=[0.16290247390873458, 0.2256891306641319,
                            0.2256891306641336, 0.09476017042552534,
                            0.6906308908961734, 0.05349939593012487,
                            0.05349939593013042, 0.08830587480616725,
                            0.0029551359803035027])
    # Ensure that we do not have excessive memory allocations 
    # (e.g., from type instabilities) 
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_weak_blast_wave_SBP.jl (Quad)" begin
    # These setups do not pass CI reliably, see
    # https://github.com/trixi-framework/Trixi.jl/pull/880 and
    # https://github.com/trixi-framework/Trixi.jl/issues/881
    @test_skip @test_trixi_include(joinpath(EXAMPLES_DIR,
                                            "elixir_mhd_weak_blast_wave_SBP.jl"),
                                   cells_per_dimension=4,
                                   # division by 2.0 corresponds to normalization by the square root of the size of the domain
                                   l2=[0.15825983698241494, 0.19897219694837923,
                                       0.19784182473275247, 0.10482833997417325,
                                       0.7310752391255246, 0.07374056714564853,
                                       0.07371172293240634, 0.10782032253431281,
                                       0.004921676235111545] ./ 2.0,
                                   linf=[0.1765644464978685, 0.2627803272865769,
                                       0.26358136695848144, 0.12347681727447984,
                                       0.7733289736898254, 0.06695360844467957,
                                       0.06650382120802623, 0.10885097000919097,
                                       0.007212567638078835])
end

@trixi_testset "elixir_mhd_weak_blast_wave_SBP.jl (Tri)" begin
    # These setups do not pass CI reliably, see
    # https://github.com/trixi-framework/Trixi.jl/pull/880 and
    # https://github.com/trixi-framework/Trixi.jl/issues/881
    @test_skip @test_trixi_include(joinpath(EXAMPLES_DIR,
                                            "elixir_mhd_weak_blast_wave_SBP.jl"),
                                   cells_per_dimension=4, element_type=Tri(),
                                   tspan=(0.0, 0.2),
                                   # division by 2.0 corresponds to normalization by the square root of the size of the domain
                                   l2=[0.13825044764021147, 0.15472815448314997,
                                       0.1549093274293255, 0.053103596213755405,
                                       0.7246162776815603, 0.07730777596615901,
                                       0.07733438386480523, 0.109893463921706,
                                       0.00617678167062838] ./ 2.0,
                                   linf=[0.22701306227317952, 0.2905255794821543,
                                       0.2912409425436937, 0.08051361477962096,
                                       1.0842974228656006, 0.07866000368926784,
                                       0.0786646354518149, 0.1614896380292925,
                                       0.010358210347485542])
end

@trixi_testset "elixir_mhd_reflective_wall.jl (Quad)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_reflective_wall.jl"),
                        cells_per_dimension=4,
                        l2=[
                            0.0036019562526881602,
                            0.0017340971255535853,
                            0.00837522167692243,
                            0.0,
                            0.028596802654003512,
                            0.0018573697892233679,
                            0.0020807798940528956,
                            0.0,
                            5.301259762428258e-5
                        ],
                        linf=[
                            0.016925983823703028,
                            0.009369659529710701,
                            0.04145170727840005,
                            0.0,
                            0.1156990108418654,
                            0.009849648257876749,
                            0.011417088537145403,
                            0.0,
                            0.0002992621756946904
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

@trixi_testset "elixir_shallowwater_source_terms.jl (Quad, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        cells_per_dimension=8, element_type=Quad(),
                        approximation_type=SBP(),
                        l2=[
                            0.0020316462913319046,
                            0.023669019044882247,
                            0.03446194752754684,
                            1.9333465252381796e-15
                        ],
                        linf=[
                            0.010385010095182778,
                            0.08750628939565086,
                            0.12088392994348407,
                            9.325873406851315e-15
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

@trixi_testset "elixir_shallowwater_source_terms.jl (Tri, SBP)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        cells_per_dimension=8, element_type=Tri(),
                        approximation_type=SBP(),
                        l2=[
                            0.004180680322490383,
                            0.07026192411558974,
                            0.11815151697006446,
                            2.329788936151192e-15
                        ],
                        linf=[
                            0.02076003852980346,
                            0.29169601664914424,
                            0.5674183379872275,
                            1.1546319456101628e-14
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

@trixi_testset "elixir_shallowwater_source_terms.jl (Tri, Polynomial)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        cells_per_dimension=8, element_type=Tri(),
                        approximation_type=Polynomial(),
                        # The last l2, linf error are the L2 projection error in approximating `b`, so they are not
                        # zero for general non-collocated quadrature rules (e.g., for `element_type=Tri()`, `polydeg > 2`).
                        l2=[
                            0.0008309356912456799,
                            0.01522451288799231,
                            0.016033969387208476,
                            1.2820247308150876e-5
                        ],
                        linf=[
                            0.001888045014140971,
                            0.05466838692127718,
                            0.06345885709961152,
                            3.3989933098554914e-5
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

@trixi_testset "elixir_shallowwater_source_terms.jl (Quad, Polynomial)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_shallowwater_source_terms.jl"),
                        cells_per_dimension=8, element_type=Quad(),
                        approximation_type=Polynomial(),
                        # The last l2, linf error are the L2 projection error in approximating `b`. However, this is zero
                        # for `Quad()` elements with `Polynomial()` approximations because the quadrature rule defaults to
                        # a `(polydeg + 1)`-point Gauss quadrature rule in each coordinate (in general, StartUpDG.jl defaults
                        # to the quadrature rule with the fewest number of points which exactly integrates the mass matrix).
                        l2=[
                            7.460461950323111e-5,
                            0.003685589808444905,
                            0.0039101604749887785,
                            2.0636891126652983e-15
                        ],
                        linf=[
                            0.000259995400729629,
                            0.0072236204211630906,
                            0.010364675200833062,
                            1.021405182655144e-14
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
end
# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn isdir(outdir) && rm(outdir, recursive = true)

end # module
