@testsnippet P4estMesh3DMHDMultiIon begin
    EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")
end

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory

@testitem "P4estMesh3D MHDMultiIon: elixir_mhdmultiion_ec.jl" setup=[
    Setup,
    P4estMesh3DMHDMultiIon
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.003734558524705688,
                            0.003860454876235077,
                            0.0038334365442637473,
                            0.002826317424003135,
                            0.002361788676521611,
                            0.002384121348044392,
                            0.002382807656697965,
                            0.038084805667035336,
                            0.003819446539691234,
                            0.0034393200529124483,
                            0.003526452713187788,
                            0.0034955536190981657,
                            0.021431701431040576,
                            5.6538192975190575e-5
                        ],
                        linf=[
                            0.2009605369841001,
                            0.17090314222579117,
                            0.19708247777146326,
                            0.0924767390720907,
                            0.09928329018905244,
                            0.10613840872102163,
                            0.11125459193977932,
                            1.204383276920653,
                            0.12581173584594696,
                            0.16978540332346442,
                            0.16748890832704827,
                            0.16866321211644908,
                            0.7017941088411468,
                            0.009214788305076223
                        ], tspan=(0.0, 0.05))
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
@testitem "P4estMesh3D MHDMultiIon: Provably entropy-stable LLF-type fluxes for multi-ion GLM-MHD" setup=[
    Setup,
    P4estMesh3DMHDMultiIon
] tags=[:p4est_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.002874086690856808,
                            0.0029624046502512006,
                            0.0029406053378946057,
                            0.0024288614580627726,
                            0.002374719116463379,
                            0.0024113028425233572,
                            0.002398986103400149,
                            0.031492698056094975,
                            0.0038750493522938757,
                            0.0032178878186358505,
                            0.0033269912777397813,
                            0.003290422650382549,
                            0.019054701919123927,
                            0.00025757334726228694
                        ],
                        linf=[
                            0.09900417586291366,
                            0.09825239764435789,
                            0.09885196003512875,
                            0.06439968749407049,
                            0.09959431539604227,
                            0.09595819250233031,
                            0.09698697827882491,
                            0.8364460019256437,
                            0.1220312744822527,
                            0.14726918166291358,
                            0.14647542854788917,
                            0.14537789842049786,
                            0.5511418902308454,
                            0.0275938439901581
                        ],
                        surface_flux=(FluxPlusDissipation(flux_ruedaramirez_etal,
                                                          DissipationLaxFriedrichsEntropyVariables(max_abs_speed_naive)),
                                      flux_nonconservative_ruedaramirez_etal),
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
