module TestExamples3DStructured

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "structured_3d_dgsem")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "StructuredMesh3D" begin
#! format: noindent

@trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
                        # Expected errors are exactly the same as with TreeMesh!
                        l2=[0.00016263963870641478],
                        linf=[0.0014537194925779984])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_free_stream.jl"),
                        l2=[1.2908196366970896e-14],
                        linf=[1.0262901639634947e-12],
                        atol=8e-13,)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_nonperiodic_curved.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_advection_nonperiodic_curved.jl"),
                        l2=[0.0004483892474201268],
                        linf=[0.009201820593762955])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
                        l2=[0.0025903889347585777],
                        linf=[0.018407576968841655],
                        # With the default `maxiters = 1` in coverage tests,
                        # there would be no time steps after the restart.
                        coverage_override=(maxiters = 100_000,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms.jl" begin
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_free_stream.jl" begin
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_free_stream.jl with FluxRotated(flux_lax_friedrichs)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
                        surface_flux=FluxRotated(flux_lax_friedrichs),
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_nonperiodic_curved.jl" begin
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_ec.jl" begin
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
                        tspan=(0.0, 0.25),
                        coverage_override=(polydeg = 3,)) # Prevent long compile time in CI
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov.jl" begin
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
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec.jl"),
                        l2=[0.009082353008355219, 0.007128360330314966,
                            0.0069703300260751545, 0.006898850266164216,
                            0.033020091335659474, 0.003203389281512797,
                            0.0030774985678369746, 0.00307400076520122,
                            4.192572922118587e-5],
                        linf=[0.28839460197220435, 0.25956437090703427,
                            0.26143649456148177, 0.24617277684934058,
                            1.1370439348603143, 0.12780410700666367,
                            0.13347392283166903,
                            0.145756208548534,
                            0.0021181795153149053],
                        tspan=(0.0, 0.25))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[0.003015390232128414, 0.0014538563096541798,
                            0.000912478356719486, 0.0017715065044433436,
                            0.0013017575272262197, 0.0014545437537522726,
                            0.0013322897333898482, 0.0016493009787844212,
                            0.0013747547738038235],
                        linf=[0.027577067632765795, 0.027912829563483885,
                            0.01282206030593043, 0.03911437990598213,
                            0.021962225923304324, 0.03169774571258743,
                            0.021591564663781426, 0.034028148178115364,
                            0.020084593242858988],
                        # Use same polydeg as everything else to prevent long compile times in CI
                        coverage_override=(polydeg = 3,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_alfven_wave.jl with flux_lax_friedrichs" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_alfven_wave.jl"),
                        l2=[0.0030477691235949685, 0.00145609137038748,
                            0.0009092809766088607, 0.0017949926915475929,
                            0.0012981612165627713, 0.0014525841626158234,
                            0.0013275465154956557, 0.0016728767532610933,
                            0.0013751925705271012],
                        linf=[0.02778552932540901, 0.027511633996169835,
                            0.012637649797178449, 0.03920805095546112,
                            0.02126543791857216, 0.031563506812970266,
                            0.02116105422516923, 0.03419432640106229,
                            0.020324891223351533],
                        surface_flux=(flux_lax_friedrichs, flux_nonconservative_powell),
                        # Use same polydeg as everything else to prevent long compile times in CI
                        coverage_override=(polydeg = 3,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhd_ec_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhd_ec_shockcapturing.jl"),
                        l2=[0.009352631216098996, 0.008058649096024162,
                            0.00802704129788766, 0.008071417834885589,
                            0.03490914976431044, 0.003930194255268652,
                            0.003921907459117296, 0.003906321239858786,
                            4.1971260184918575e-5],
                        linf=[0.307491045404509, 0.26790087991041506,
                            0.2712430701672931, 0.2654540237991884,
                            0.9620943261873176, 0.181632512204141,
                            0.15995711137712265, 0.1791807940466812,
                            0.015138421396338456],
                        tspan=(0.0, 0.25),
                        # Use same polydeg as everything else to prevent long compile times in CI
                        coverage_override=(polydeg = 3,))
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
@test_nowarn rm(outdir, recursive = true)

end # module
