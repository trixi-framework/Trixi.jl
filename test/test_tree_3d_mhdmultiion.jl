module TestExamples3DIdealGlmMhdMultiIon

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(examples_dir(), "tree_3d_dgsem")

@testset "MHD Multi-ion" begin
#! format: noindent

@trixi_testset "elixir_mhdmultiion_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.005515087802594469,
                            0.00551509073876885,
                            0.005515594555021946,
                            0.0034090003657850213,
                            0.0037098074865083786,
                            0.003709809008606359,
                            0.003709945218162014,
                            0.04983950169154286,
                            0.005484133811034023,
                            0.0052829327369328825,
                            0.0052829303233862075,
                            0.005283547653931791,
                            0.033527917281977085,
                            3.745797136118132e-5
                        ],
                        linf=[
                            0.21015619845899525,
                            0.21016594468652272,
                            0.21016372645728998,
                            0.07639539286669705,
                            0.1563224943342573,
                            0.15632897896625103,
                            0.15634118799842192,
                            1.0389697085654417,
                            0.22150196712181291,
                            0.2513260575632066,
                            0.2513406247910507,
                            0.2513195899676654,
                            0.850469838195949,
                            0.007393577064900667
                        ], tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
@trixi_testset "Provably entropy-stable LLF-type fluxes for multi-ion GLM-MHD" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.005460794624649135,
                            0.005460794781236467,
                            0.0054609085902993714,
                            0.003401374056052151,
                            0.0036208457535567657,
                            0.003620845818195209,
                            0.003620858239938361,
                            0.04989063985215975,
                            0.005418705706275181,
                            0.0051299807116148306,
                            0.0051299815562970644,
                            0.005130087742236932,
                            0.03343342218681378,
                            0.0003014824724440717
                        ],
                        linf=[
                            0.1218555555063573,
                            0.12182377208983375,
                            0.1272962683915283,
                            0.07184697318740407,
                            0.1290519098461309,
                            0.1290415359815689,
                            0.12903793013111556,
                            1.0816799191698845,
                            0.17564429852964403,
                            0.16744331019561018,
                            0.16743123506849708,
                            0.16741878260454496,
                            0.9586779761175728,
                            0.04959304829601756
                        ],
                        surface_flux=(FluxPlusDissipation(flux_ruedaramirez_etal,
                                                          DissipationLaxFriedrichsEntropyVariables(max_abs_speed_naive)),
                                      flux_nonconservative_ruedaramirez_etal),
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

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
@trixi_testset "elixir_mhdmultiion_ec.jl with local Lax-Friedrichs at the surface" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.005460798875980804,
                            0.005460799031160416,
                            0.005460913046698877,
                            0.0034013619963554655,
                            0.003620841554741119,
                            0.003620841620616253,
                            0.003620853848969877,
                            0.049890563484378106,
                            0.005418681699884455,
                            0.00512998900051011,
                            0.005129989844094721,
                            0.005130096170780701,
                            0.033433442667615464,
                            0.0003014502208853599
                        ],
                        linf=[
                            0.1219052420427541,
                            0.12187345128461624,
                            0.12736602309528855,
                            0.0718706843831256,
                            0.1290799868829374,
                            0.12906962204819608,
                            0.1290662650327416,
                            1.0816988809951793,
                            0.17571730225132676,
                            0.1674224916296424,
                            0.16741032857524493,
                            0.16739786812044885,
                            0.9587397297591731,
                            0.04960518070421052
                        ],
                        surface_flux=(FluxLaxFriedrichs(max_abs_speed_naive),
                                      flux_nonconservative_central),
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
end

end # module
