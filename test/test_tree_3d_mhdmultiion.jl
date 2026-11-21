@testsnippet TreeMesh3DMHDMultiIon begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_3d_dgsem")
end

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory

@testitem "TreeMesh3D MHDMultiIon: elixir_mhdmultiion_ec.jl" setup=[
    Setup,
    TreeMesh3DMHDMultiIon
] tags=[:tree_part5] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.004956468192216787,
                            0.004956464556149274,
                            0.004959906868699944,
                            0.0029980839075595634,
                            0.0030632400851144273,
                            0.003063241615786391,
                            0.0030622442594148112,
                            0.04329049623791101,
                            0.004971844621279697,
                            0.004205460432376655,
                            0.004205485085995445,
                            0.004205542275763168,
                            0.02970702883947866,
                            2.6365712825520774e-5
                        ],
                        linf=[
                            0.2167876930511824,
                            0.21678739728212715,
                            0.2208567346296907,
                            0.07053210259446119,
                            0.12099008463655332,
                            0.12102090712626135,
                            0.12183228575911499,
                            1.0420503523829785,
                            0.18100932441443562,
                            0.2098606235095619,
                            0.21003309683648907,
                            0.21286882300849005,
                            0.8672319702871572,
                            0.007147352526199665
                        ], tspan=(0.0, 0.05),
                        initial_refinement_level=3)
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
@testitem "TreeMesh3D MHDMultiIon: Provably entropy-stable LLF-type fluxes for multi-ion GLM-MHD" setup=[
    Setup,
    TreeMesh3DMHDMultiIon
] tags=[:tree_part5] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.004775006343502511,
                            0.004775005434282501,
                            0.004776366443934422,
                            0.0028373538532592995,
                            0.0028892615736531,
                            0.0028892604412767337,
                            0.0028894838040734497,
                            0.04124170066836797,
                            0.0048844850999567695,
                            0.00385922446798913,
                            0.0038592311207969483,
                            0.0038599711827878467,
                            0.029040521507746664,
                            0.00023086288729964218
                        ],
                        linf=[
                            0.1376180072055817,
                            0.13758502948734375,
                            0.1458389217851389,
                            0.07926622510788506,
                            0.09842466633606323,
                            0.09842683320602831,
                            0.09855058946777351,
                            1.2069466743694282,
                            0.17195992671333427,
                            0.15095824407738148,
                            0.15108959578203224,
                            0.1677206256847408,
                            0.9898735086895312,
                            0.032470364483618126
                        ],
                        surface_flux=(FluxPlusDissipation(flux_ruedaramirez_etal,
                                                          DissipationLaxFriedrichsEntropyVariables(max_abs_speed_naive)),
                                      flux_nonconservative_ruedaramirez_etal),
                        tspan=(0.0, 0.05),
                        initial_refinement_level=3)
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
@testitem "TreeMesh3D MHDMultiIon: elixir_mhdmultiion_ec.jl with local Lax-Friedrichs at the surface" setup=[
    Setup,
    TreeMesh3DMHDMultiIon
] tags=[:tree_part5] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.004774942499373016,
                            0.004774941592808391,
                            0.004776300650427028,
                            0.0028370364607463797,
                            0.002888931106360228,
                            0.0028889299870399573,
                            0.002889159504929624,
                            0.04123815403604294,
                            0.004884355597988775,
                            0.0038588259444425708,
                            0.0038588325547317336,
                            0.0038595840000014234,
                            0.029039740440643964,
                            0.00023094169413139285
                        ],
                        linf=[
                            0.13758994679584502,
                            0.13755730930579524,
                            0.14565695008677304,
                            0.07930723942201318,
                            0.09836516126228151,
                            0.09836733130117514,
                            0.09849054848280701,
                            1.2073728371412513,
                            0.17197397970795258,
                            0.15084523348961176,
                            0.15097594158645608,
                            0.16767239923726596,
                            0.989720887334371,
                            0.032494627702930594
                        ],
                        surface_flux=(FluxLaxFriedrichs(max_abs_speed_naive),
                                      flux_nonconservative_central),
                        tspan=(0.0, 0.05),
                        initial_refinement_level=3)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
