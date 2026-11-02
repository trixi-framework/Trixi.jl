@testsnippet StructuredMesh3D begin
    EXAMPLES_DIR = joinpath(examples_dir(), "structured_3d_dgsem")
end

@testitem "StructuredMesh3D: elixir_advection_basic.jl" setup=[Setup, StructuredMesh3D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_advection_free_stream.jl" setup=[
    Setup,
    StructuredMesh3D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream.jl"),
                        l2=[1.2908196366970896e-14],
                        linf=[1.0262901639634947e-12],
                        atol=8e-13)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_advection_nonperiodic_curved.jl" setup=[
    Setup,
    StructuredMesh3D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_nonperiodic_curved.jl"),
                        l2=[0.0004483892474201268],
                        linf=[0.009201820593762955])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_advection_restart.jl" setup=[Setup, StructuredMesh3D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[0.0025903889347585777],
                        linf=[0.018407576968841655])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_euler_source_terms.jl" setup=[Setup, StructuredMesh3D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[
                            0.010385936842224346,
                            0.009776048833895767,
                            0.00977604883389591,
                            0.009776048833895733,
                            0.01506687097416608
                        ],
                        linf=[
                            0.03285848350791731,
                            0.0321792316408982,
                            0.032179231640894645,
                            0.032179231640895534,
                            0.0655408023333299
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_euler_free_stream.jl" setup=[Setup, StructuredMesh3D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        l2=[
                            2.8815700334367128e-15,
                            9.361915278236651e-15,
                            9.95614203619935e-15,
                            1.6809941842374106e-14,
                            1.4815037041566735e-14
                        ],
                        linf=[
                            4.1300296516055823e-14,
                            2.0444756998472258e-13,
                            1.0133560657266116e-13,
                            2.0627943797535409e-13,
                            2.8954616482224083e-13
                        ])
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
@testitem "StructuredMesh3D: elixir_euler_free_stream.jl with FluxRotated(FluxLaxFriedrichs(max_abs_speed_naive))" setup=[
    Setup,
    StructuredMesh3D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        surface_flux=FluxRotated(FluxLaxFriedrichs(max_abs_speed_naive)),
                        l2=[
                            2.8815700334367128e-15,
                            9.361915278236651e-15,
                            9.95614203619935e-15,
                            1.6809941842374106e-14,
                            1.4815037041566735e-14
                        ],
                        linf=[
                            4.1300296516055823e-14,
                            2.0444756998472258e-13,
                            1.0133560657266116e-13,
                            2.0627943797535409e-13,
                            2.8954616482224083e-13
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_euler_source_terms_nonperiodic_curved.jl" setup=[
    Setup,
    StructuredMesh3D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic_curved.jl"),
                        l2=[
                            0.0032940531178824463,
                            0.003275679548217804,
                            0.0030020672748714084,
                            0.00324007343451744,
                            0.005721986362580164
                        ],
                        linf=[
                            0.03156756290660656,
                            0.033597629023726316,
                            0.02095783702361409,
                            0.03353574465232212,
                            0.05873635745032857
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_euler_source_terms_adaptive_vol_int.jl" setup=[
    Setup,
    StructuredMesh3D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_adaptive_vol_int.jl"),
                        l2=[
                            0.0022878466025464924,
                            0.0022468985135237107,
                            0.002033101904326531,
                            0.0022483293238877004,
                            0.004345885176536526
                        ],
                        linf=[
                            0.02684833550245891,
                            0.028038458002909383,
                            0.022367626384228956,
                            0.02930352451545737,
                            0.05222743788433171
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_euler_source_terms_nonperiodic_fvO2.jl" setup=[
    Setup,
    StructuredMesh3D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic_fvO2.jl"),
                        l2=[
                            0.046860892952192236,
                            0.04269641872975366,
                            0.04269641872975368,
                            0.042696418729753556,
                            0.1443421265167028
                        ],
                        linf=[
                            0.42250001347847244,
                            0.2975151811754566,
                            0.29751518117545483,
                            0.29751518117545617,
                            0.3982472383589144
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_euler_ec.jl" setup=[Setup, StructuredMesh3D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.011367083018614027,
                            0.007022020327490176,
                            0.006759580335962235,
                            0.006820337637760632,
                            0.02912659127566544
                        ],
                        linf=[
                            0.2761764220925329,
                            0.20286331858055706,
                            0.18763944865434593,
                            0.19313636558790004,
                            0.707563913727584
                        ],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_euler_sedov.jl" setup=[Setup, StructuredMesh3D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
                        l2=[
                            5.30310390e-02,
                            2.53167260e-02,
                            2.64276438e-02,
                            2.52195992e-02,
                            3.56830295e-01
                        ],
                        linf=[
                            6.16356950e-01,
                            2.50600049e-01,
                            2.74796377e-01,
                            2.46448217e-01,
                            4.77888479e+00
                        ],
                        tspan=(0.0, 0.3))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_mhd_ec.jl" setup=[Setup, StructuredMesh3D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
                        l2=[0.009082686863634625, 0.007127253247814603,
                            0.006969252837723283, 0.0068981421428559175,
                            0.03299141065201546, 0.0032090380835872174,
                            0.003073657491990538, 0.0030775450911821214,
                            0.00016399139349350833],
                        linf=[0.28822490352996655, 0.25936033575115947,
                            0.2613686580019892, 0.24609839138134065,
                            1.131964694567725, 0.1259190764818081,
                            0.13403361588095375,
                            0.14751044288572324,
                            0.007122681079015943],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_mhd_alfven_wave.jl" setup=[Setup, StructuredMesh3D] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[0.0029696257394729152, 0.0012622216030528468,
                            0.0008671267653445159, 0.001564357917158349,
                            0.001113546942959469, 0.001018538178276083,
                            0.0007016185553470859, 0.001503673511544824,
                            0.0004870160708328902],
                        linf=[0.028141458589278634, 0.0181627267215195,
                            0.012201308039418393, 0.03144969126150016,
                            0.015648797337592057, 0.01768874352408012,
                            0.008920488167988183, 0.027122186873299793,
                            0.007736308004186873])
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
@testitem "StructuredMesh3D: elixir_mhd_alfven_wave.jl with FluxLaxFriedrichs(max_abs_speed_naive)" setup=[
    Setup,
    StructuredMesh3D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[0.0029910047917099066, 0.001265048291892031,
                            0.0008658699687391328, 0.001589544836512516,
                            0.0011070545379741697, 0.0010191783277482412,
                            0.0006972098973368367, 0.0015265352357127868,
                            0.0004854003243287278],
                        linf=[0.028004484929964435, 0.01821369996766581,
                            0.012254504590014356, 0.03164394520483074,
                            0.01549053812453538, 0.017693782416169923,
                            0.008736916755356594, 0.027560376655571046,
                            0.007704016295357286],
                        surface_flux=(FluxLaxFriedrichs(max_abs_speed_naive),
                                      flux_nonconservative_powell))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "StructuredMesh3D: elixir_mhd_ec_shockcapturing.jl" setup=[
    Setup,
    StructuredMesh3D
] tags=[:structured] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec_shockcapturing.jl"),
                        l2=[0.009350667580083096, 0.008057221618232135,
                            0.00802601147264494, 0.008071669939895765,
                            0.03494353248740231, 0.003929251134703461,
                            0.003924057092861661, 0.00390790726383187,
                            0.0002585910825656062],
                        linf=[0.30715472775434094, 0.2678401862912938,
                            0.27397404386819235, 0.2639774339612127,
                            0.9841410014301148, 0.18629819425753347,
                            0.15396669659676487, 0.17480399289034199,
                            0.0308043062713562],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
