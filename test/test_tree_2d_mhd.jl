module TestExamples2DMHD

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "MHD" begin
#! format: noindent

@trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            0.00011149543672225127,
                            5.888242524520296e-6,
                            5.888242524510072e-6,
                            8.476931432519067e-6,
                            1.3160738644036652e-6,
                            1.2542675002588144e-6,
                            1.2542675002747718e-6,
                            1.8705223407238346e-6,
                            4.651717010670585e-7
                        ],
                        linf=[
                            0.00026806333988971254,
                            1.6278838272418272e-5,
                            1.627883827305665e-5,
                            2.7551183488072617e-5,
                            5.457878055614707e-6,
                            8.130129322880819e-6,
                            8.130129322769797e-6,
                            1.2406302192291552e-5,
                            2.373765544951732e-6
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

@trixi_testset "elixir_mhd_alfven_wave.jl with flux_derigs_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            1.7201098719531215e-6,
                            8.692057393373005e-7,
                            8.69205739320643e-7,
                            1.2726508184718958e-6,
                            1.040607127595208e-6,
                            1.07029565814218e-6,
                            1.0702956581404748e-6,
                            1.3291748105236525e-6,
                            4.6172239295786824e-7
                        ],
                        linf=[
                            9.865325754310206e-6,
                            7.352074675170961e-6,
                            7.352074674185638e-6,
                            1.0675656902672803e-5,
                            5.112498347226158e-6,
                            7.789533065905019e-6,
                            7.789533065905019e-6,
                            1.0933531593274037e-5,
                            2.340244047768378e-6
                        ],
                        volume_flux=(flux_derigs_etal, flux_nonconservative_powell))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave_mortar.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
                        l2=[
                            1.0896015330565795e-5,
                            4.152763046029908e-6,
                            3.851874655132384e-6,
                            4.2295110232831874e-6,
                            3.135859402264645e-6,
                            3.29531401471973e-6,
                            3.1347238307092746e-6,
                            4.186230495566739e-6,
                            1.670859989962532e-6
                        ],
                        linf=[
                            5.3178641410078775e-5,
                            3.09217107711951e-5,
                            2.7722788709688695e-5,
                            2.1631700804783383e-5,
                            1.558520409727926e-5,
                            1.73873627985488e-5,
                            1.6635747942750356e-5,
                            2.0751205947883156e-5,
                            7.655540399230342e-6
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

@trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
                        l2=[
                            0.03637302248881514,
                            0.043002991956758996,
                            0.042987505670836056,
                            0.02574718055258975,
                            0.1621856170457943,
                            0.01745369341302589,
                            0.017454552320664566,
                            0.026873190440613117,
                            5.336243933079389e-16
                        ],
                        linf=[
                            0.23623816236321427,
                            0.3137152204179957,
                            0.30378397831730597,
                            0.21500228807094865,
                            0.9042495730546518,
                            0.09398098096581875,
                            0.09470282020962917,
                            0.15277253978297378,
                            4.307694418935709e-15
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

@trixi_testset "elixir_mhd_ec_float32.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec_float32.jl"),
                        l2=Float32[0.03635566,
                                   0.042947732,
                                   0.042947736,
                                   0.025748001,
                                   0.16211228,
                                   0.01745248,
                                   0.017452491,
                                   0.026877586,
                                   2.417227f-7],
                        linf=Float32[0.2210092,
                                     0.28798974,
                                     0.28799006,
                                     0.20858109,
                                     0.8812673,
                                     0.09208107,
                                     0.09208131,
                                     0.14795369,
                                     2.2078211f-6])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_orszag_tang.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
                        l2=[
                            0.21970081242543155,
                            0.2643108041773596,
                            0.31484243079966445,
                            0.0,
                            0.5159994161306146,
                            0.23024218609799854,
                            0.34413704351228147,
                            0.0,
                            0.003220120866497733
                        ],
                        linf=[
                            1.2753954566712156,
                            0.6737923290533722,
                            0.8574465081172007,
                            0.0,
                            2.800507621357904,
                            0.6472414758680339,
                            0.9707631523292184,
                            0.0,
                            0.06528658804650658
                        ],
                        tspan=(0.0, 0.09))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_orszag_tang.jl with flux_hlle" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
                        l2=[
                            0.10806640059794005,
                            0.20199169830949384,
                            0.22984626162122923,
                            0.0,
                            0.2995035634728381,
                            0.1568851137962238,
                            0.24293639539810255,
                            0.0,
                            0.003246131507524401
                        ],
                        linf=[
                            0.5600698267839397,
                            0.5095520220558266,
                            0.6536747859174317,
                            0.0,
                            0.9624343226044095,
                            0.39814285051228965,
                            0.6734722065677001,
                            0.0,
                            0.048789764358224214
                        ],
                        tspan=(0.0, 0.06),
                        surface_flux=(flux_hlle,
                                      flux_nonconservative_powell))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl one step with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[
                            7.144325530681224e-17,
                            2.123397983547417e-16,
                            5.061138912500049e-16,
                            3.6588423152083e-17,
                            8.449816179702522e-15,
                            3.9171737639099993e-16,
                            2.445565690318772e-16,
                            3.6588423152083e-17,
                            9.971153407737885e-17
                        ],
                        linf=[
                            2.220446049250313e-16,
                            8.465450562766819e-16,
                            1.8318679906315083e-15,
                            1.1102230246251565e-16,
                            1.4210854715202004e-14,
                            8.881784197001252e-16,
                            4.440892098500626e-16,
                            1.1102230246251565e-16,
                            4.779017148551244e-16
                        ],
                        maxiters=1,
                        initial_condition=initial_condition_constant,
                        atol=2.0e-13)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
                        l2=[
                            1.264189543599029,
                            1.8320601832078407,
                            1.700447583152504,
                            0.0,
                            2.3024199507805165,
                            0.21477383173627232,
                            0.23559923070707714,
                            0.0,
                            0.0034025828879598176
                        ],
                        linf=[
                            10.988505627764773,
                            14.712395261659752,
                            15.687199838635722,
                            0.0,
                            17.095921959435447,
                            1.335014119480973,
                            1.4366904817630641,
                            0.0,
                            0.08464617851256993
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

@trixi_testset "elixir_mhd_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave.jl"),
                        l2=[
                            0.17638656371490055,
                            3.8616031530927084,
                            2.4810236453809127,
                            0.0,
                            354.6341111396657,
                            2.353681534580767,
                            1.3926633033090652,
                            0.0,
                            0.030696738560246576
                        ],
                        linf=[
                            1.5823311254590813,
                            44.156859286717044,
                            13.036736942960012,
                            0.0,
                            2187.5906984085345,
                            12.552321899505023,
                            9.147117303057248,
                            0.0,
                            0.5285917066723818
                        ],
                        tspan=(0.0, 0.003),)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_shockcapturing_subcell.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_shockcapturing_subcell.jl"),
                        l2=[
                            3.2064026219236076e-02,
                            7.2461094392606618e-02,
                            7.2380202888062711e-02,
                            0.0000000000000000e+00,
                            8.6293936673145932e-01,
                            8.4091669534557805e-03,
                            5.2156364913231732e-03,
                            0.0000000000000000e+00,
                            2.0786952301129021e-04
                        ],
                        linf=[
                            3.8778760255775635e-01,
                            9.4666683953698927e-01,
                            9.4618924645661928e-01,
                            0.0000000000000000e+00,
                            1.0980297261521951e+01,
                            1.0264404591009069e-01,
                            1.0655686942176350e-01,
                            0.0000000000000000e+00,
                            6.1013422157115546e-03
                        ],
                        tspan=(0.0, 0.003))
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
end

end # module
