@testsnippet TreeMesh2DMHD begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")
end

@testitem "TreeMesh2D MHD: elixir_mhd_alfven_wave.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
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
    # Test `show()`
    @trixi_test_nowarn show(IOContext(stdout, :compact => false), glm_speed_callback)

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_alfven_wave.jl with flux_derigs_etal" setup=[
    Setup,
    TreeMesh2DMHD
] tags=[:tree_part3] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_alfven_wave_dirichlet.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_dirichlet.jl"),
                        l2=[
                            0.00011004538877483271,
                            5.926645116290825e-6,
                            5.931933718790244e-6,
                            8.482384248361835e-6,
                            1.4150070042287573e-6,
                            1.3803265179621126e-6,
                            1.373512939846543e-6,
                            2.2630780221312974e-6,
                            8.186309170400813e-7
                        ],
                        linf=[
                            0.0002801680361144143,
                            1.8417644682994228e-5,
                            1.8537339994670332e-5,
                            3.0471153402808482e-5,
                            8.604473854645356e-6,
                            1.0055681487042278e-5,
                            1.0191055124897375e-5,
                            1.7660034751995624e-5,
                            4.141061665063549e-6
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_alfven_wave_mortar.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave_mortar.jl"),
                        l2=[
                            1.7817867247272455e-6,
                            8.788054707093757e-7,
                            8.596640315610475e-7,
                            1.2487243970329486e-6,
                            1.0434652347305932e-6,
                            9.72009633342213e-7,
                            9.729248512929134e-7,
                            1.3240721048567003e-6,
                            4.710099772228978e-7
                        ],
                        linf=[
                            2.9664250134953107e-5,
                            9.05296547021317e-6,
                            8.777227916395569e-6,
                            1.2503499413729635e-5,
                            7.90893283519889e-6,
                            8.92383335737712e-6,
                            8.589000874859032e-6,
                            1.2108689799256167e-5,
                            4.649918116658749e-6
                        ],
                        tspan=(0.0, 1.0))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_ec.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_ec_float32.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
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
                                     2.2078211f-6],
                        RealT_for_test_tolerances=Float32)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_orszag_tang.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
                        l2=[
                            0.2196896072611031,
                            0.2643149625630577,
                            0.31492071537345334,
                            0.0,
                            0.5160634025012188,
                            0.23035912757434615,
                            0.34414766819550974,
                            0.0,
                            0.00312416589095812
                        ],
                        linf=[
                            1.2746442244018454,
                            0.6734738397287374,
                            0.8631029321589829,
                            0.0,
                            2.7985233284833875,
                            0.6525571599049463,
                            0.967392391469731,
                            0.0,
                            0.05705437906440683
                        ],
                        tspan=(0.0, 0.09))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_orszag_tang.jl with flux_hlle" setup=[
    Setup,
    TreeMesh2DMHD
] tags=[:tree_part3] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_orszag_tang.jl"),
                        l2=[
                            0.10806609425776458,
                            0.20199108727890655,
                            0.22984592632472103,
                            0.0,
                            0.2994998927864948,
                            0.15688280889599093,
                            0.24293668114219336,
                            0.0,
                            0.003245200281656975
                        ],
                        linf=[
                            0.5604737569203582,
                            0.5095520220558266,
                            0.6536758568490424,
                            0.0,
                            0.96319026400795,
                            0.3981344244658036,
                            0.6734721166641753,
                            0.0,
                            0.04878941836831004
                        ],
                        tspan=(0.0, 0.06),
                        surface_flux=(flux_hlle,
                                      flux_nonconservative_powell))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_alfven_wave.jl one step with initial_condition_constant" setup=[
    Setup,
    TreeMesh2DMHD
] tags=[:tree_part3] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_rotor.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_rotor.jl"),
                        l2=[
                            1.2598402656899168,
                            1.820971929991119,
                            1.6999644960527263,
                            0.0,
                            2.292864175916754,
                            0.21457609873393194,
                            0.2358014822442555,
                            0.0,
                            0.0031149245799970845
                        ],
                        linf=[
                            11.01924588382204,
                            14.60347214527161,
                            15.67021495738901,
                            0.0,
                            17.095764983758468,
                            1.326211734209035,
                            1.4362411285277645,
                            0.0,
                            0.08275874670689418
                        ],
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_blast_wave.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_blast_wave.jl"),
                        l2=[
                            0.17649071751321999,
                            3.866334873356359,
                            2.4884369702456235,
                            0.0,
                            355.34561816675614,
                            2.3558717566361573,
                            1.4050655117370945,
                            0.0,
                            0.02778107563387455
                        ],
                        linf=[
                            1.5806072087974057,
                            44.14797256569522,
                            13.072016885780329,
                            0.0,
                            2245.1852927408295,
                            13.075560456737785,
                            9.147335384774467,
                            0.0,
                            0.5109516424638774
                        ],
                        tspan=(0.0, 0.003))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_shockcapturing_subcell.jl" setup=[
    Setup,
    TreeMesh2DMHD
] tags=[:tree_part3] begin
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15000)
end

# This is tested with reference values for the local-symmetric formulation.
@testitem "TreeMesh2D MHD: elixir_mhd_shockcapturing_subcell.jl (local-jump formulation)" setup=[
    Setup,
    TreeMesh2DMHD
] tags=[:tree_part3] begin
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
                        tspan=(0.0, 0.003),
                        # Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
                        # `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
                        # In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
                        # Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
                        # To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
                        # We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
                        # `StepsizeCallback` (CFL-Condition) and less diffusion.
                        surface_flux=(FluxLaxFriedrichs(max_abs_speed_naive),
                                      flux_nonconservative_powell_local_jump),
                        volume_flux=(flux_derigs_etal,
                                     flux_nonconservative_powell_local_jump))
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
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15000)
end

@testitem "TreeMesh2D MHD: elixir_mhd_onion.jl" setup=[Setup, TreeMesh2DMHD] tags=[:tree_part3] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_onion.jl"),
                        l2=[
                            0.006145640007814805,
                            0.04298975802036206,
                            0.009442308958879332,
                            0.0,
                            0.023466074687332656,
                            0.0037008482451226085,
                            0.006939946291086656,
                            0.0,
                            2.6526292145616063e-6
                        ],
                        linf=[
                            0.04034000344526789,
                            0.25073951149407886,
                            0.05597857594719327,
                            0.0,
                            0.14115800038105397,
                            0.019956735193905117,
                            0.03867389126521381,
                            0.0,
                            2.168681783467941e-5
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
