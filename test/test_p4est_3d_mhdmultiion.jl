module TestExamples3DIdealGlmMhdMultiIon

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(examples_dir(), "p4est_3d_dgsem")

@testset "MHD Multi-ion" begin
#! format: noindent

@trixi_testset "elixir_mhdmultiion_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.003734820532985263,
                            0.0038615619955546204,
                            0.003833638703487458,
                            0.0028263125135739955,
                            0.0023618139630058143,
                            0.002384129259605738,
                            0.0023828293292904833,
                            0.038089526935383124,
                            0.0038194562843790105,
                            0.003439479296467246,
                            0.003526498173885533,
                            0.0034956719160693537,
                            0.02143679204123428,
                            3.3199722425501164e-6
                        ],
                        linf=[
                            0.2016143602393723,
                            0.1707562492816741,
                            0.19671841540041113,
                            0.092481456884773,
                            0.09928141143714853,
                            0.10613515319792097,
                            0.11127460598498372,
                            1.2130363801029604,
                            0.12582249707043758,
                            0.1698494562737311,
                            0.16751667624425207,
                            0.1687325700572586,
                            0.7019146966991214,
                            0.0007812116161561696
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
                            0.0028955650582195335,
                            0.0029749524273695624,
                            0.0029533907732692344,
                            0.0024285755843305988,
                            0.0023749924019966094,
                            0.0024114801452486206,
                            0.0023991164504279404,
                            0.03150571433653016,
                            0.003874135295085382,
                            0.0032206538891184646,
                            0.0033289635387358306,
                            0.0032922608966505377,
                            0.019053919178522397,
                            1.2888164218472409e-5
                        ],
                        linf=[
                            0.10031944778984792,
                            0.09835765892858706,
                            0.09915484563347321,
                            0.0643535254948433,
                            0.09957773301344566,
                            0.09607387748333969,
                            0.09698728400727108,
                            0.8341002490873852,
                            0.12157560398831846,
                            0.14778536942358805,
                            0.1464940331696904,
                            0.1449618481727096,
                            0.5487673957733081,
                            0.0014540668676888365
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

@trixi_testset "elixir_mhdmultiion_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_convergence.jl"),
                        l2=[
                            2.7007694451840977e-5,
                            2.252632596997783e-5,
                            1.830892822103072e-5,
                            1.7457386132678724e-5,
                            3.965825276181703e-5,
                            6.886878771068099e-5,
                            3.216774733720572e-5,
                            0.00013796601797391608,
                            2.762642533644496e-5,
                            7.877500410069398e-5,
                            0.00012184040930856932,
                            8.918795955887214e-5,
                            0.0002122739932637704,
                            1.0532691581216071e-6
                        ],
                        linf=[
                            0.0005846835977684206,
                            0.00031591380039502903,
                            0.0002529555339790268,
                            0.0003873459403432866,
                            0.0007355557980894822,
                            0.0012929706727252688,
                            0.0002558003707378437,
                            0.0028085112041740246,
                            0.0006114366794293113,
                            0.001257825301983151,
                            0.0018924211424776738,
                            0.0007347447431757664,
                            0.004148291057411768,
                            1.8948511576480304e-5
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
end
end # module
