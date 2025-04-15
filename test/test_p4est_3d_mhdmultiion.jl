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
                                                          DissipationLaxFriedrichsEntropyVariables()),
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
end
end # module
