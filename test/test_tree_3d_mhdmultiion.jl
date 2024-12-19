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
                            0.005515090292575059,
                            0.005515093229701533,
                            0.0055155968594217,
                            0.0034090002245163614,
                            0.003709807395174228,
                            0.003709808917203165,
                            0.003709945123475921,
                            0.04983943937107913,
                            0.005484133454336887,
                            0.00528293290439966,
                            0.005282930490865487,
                            0.005283547806305909,
                            0.03352789135708704,
                            3.749645231098896e-5
                        ],
                        linf=[
                            0.21024185596950629,
                            0.21025151478760273,
                            0.21024912618268798,
                            0.076395739096276,
                            0.1563223611743002,
                            0.15632884538092198,
                            0.15634105391848113,
                            1.0388823226534836,
                            0.22150908356656462,
                            0.2513262952807552,
                            0.2513408613352427,
                            0.25131982443776496,
                            0.8504449549199755,
                            0.00745904702407851
                        ], tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Slightly larger values for allowed allocations due to the size of the multi-ion MHD system
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 2000
    end
end

@trixi_testset "Provably entropy-stable LLF-type fluxes for multi-ion GLM-MHD" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.005461797766886852,
                            0.0054617979185473935,
                            0.005461911963493437,
                            0.0034019650432160183,
                            0.0036214136925153332,
                            0.003621413755863293,
                            0.0036214256331343125,
                            0.04989956402093448,
                            0.005418917526099802,
                            0.005131066757973498,
                            0.005131067607695321,
                            0.005131172274784049,
                            0.03343666846482734,
                            0.0003007804111764196
                        ],
                        linf=[
                            0.12234892772192096,
                            0.1223171830655686,
                            0.1278258253016613,
                            0.07205879616765087,
                            0.1290061932668167,
                            0.12899478253220256,
                            0.12899463311942488,
                            1.084354847523644,
                            0.17599973652527756,
                            0.1673024753470816,
                            0.16729366599933276,
                            0.1672749128649947,
                            0.9610489935515938,
                            0.04894654146236637
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
        # Slightly larger values for allowed allocations due to the size of the multi-ion MHD system
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 2000
    end
end

@trixi_testset "elixir_mhdmultiion_ec.jl with local Lax-Friedrichs at the surface" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmultiion_ec.jl"),
                        l2=[
                            0.0054618017243060245,
                            0.005461801874647487,
                            0.005461916115692502,
                            0.0034019533360683516,
                            0.0036214103139306027,
                            0.003621410378506676,
                            0.0036214220691497454,
                            0.04989948890811091,
                            0.005418894551414051,
                            0.005131075328225897,
                            0.005131076176988492,
                            0.005131180975100698,
                            0.033436687541996704,
                            0.00030074542007975337
                        ],
                        linf=[
                            0.12239622025605945,
                            0.12236446866852024,
                            0.12789210451104582,
                            0.07208147196589526,
                            0.12903267474777533,
                            0.12902126950883053,
                            0.12902139991772607,
                            1.0843706600615892,
                            0.1760697478878981,
                            0.16728192094436928,
                            0.16727303109462638,
                            0.1672542060574919,
                            0.9611095787397885,
                            0.04895673952280341
                        ],
                        surface_flux=(flux_lax_friedrichs, flux_nonconservative_central),
                        tspan=(0.0, 0.05))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        # Slightly larger values for allowed allocations due to the size of the multi-ion MHD system
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 2000
    end
end
end

end # module
