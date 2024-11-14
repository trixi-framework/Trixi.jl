module TestExamples2DEulerMulticomponent

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_2d_dgsem")

@testset "MHD Multicomponent" begin
#! format: noindent

@trixi_testset "elixir_mhdmulti_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
                        l2=[0.04300299195675897, 0.042987505670835945,
                            0.025747180552589767, 0.1621856170457937,
                            0.017453693413025828, 0.0174545523206645,
                            0.026873190440613162, 1.364647699274761e-15,
                            0.012124340829605002, 0.024248681659210004],
                        linf=[0.31371522041799105, 0.3037839783173047,
                            0.21500228807094351, 0.904249573054642,
                            0.0939809809658183, 0.09470282020962761, 0.1527725397829759,
                            8.245701827530042e-15,
                            0.0787460541210726, 0.1574921082421452])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhdmulti_ec.jl with flux_derigs_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_ec.jl"),
                        l2=[0.04301155595653799, 0.04299735787276207,
                            0.025745530869947714,
                            0.16206102676791553, 0.017454384272339165,
                            0.01745523378100091,
                            0.026879482381500154, 0.0002038008756963954,
                            0.012094208262809778,
                            0.024188416525619556],
                        linf=[0.3156206778985397, 0.30941696929809526,
                            0.21167563519254176,
                            0.9688251298546122, 0.09076254289155083,
                            0.09160589769498295,
                            0.15698032974768705, 0.006131914796912965,
                            0.07839287555951036,
                            0.1567857511190207],
                        volume_flux=(flux_derigs_etal, flux_nonconservative_powell),
                        surface_flux=(flux_derigs_etal, flux_nonconservative_powell))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhdmulti_es.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_es.jl"),
                        l2=[0.042511527162267, 0.04250603277530184, 0.02385422747993974,
                            0.11555081362726903,
                            0.016366641053738043, 0.01636681584592762,
                            0.02581748418797907, 0.00023394429554818215,
                            0.010834603551662698, 0.021669207103325396],
                        linf=[0.23454607703107877, 0.23464789247380322,
                            0.11898832084115452, 0.5331209602648022,
                            0.061744814466827336, 0.061767127585091286,
                            0.09595041452184983, 0.004421037168524759,
                            0.06186597801911198, 0.12373195603822396])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhdmulti_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_convergence.jl"),
                        l2=[0.0003808877028249613, 0.0003808877028249593,
                            0.0005155994511260122, 0.000570394227652563,
                            0.000439568811048544, 0.0004395688110485541,
                            0.0005074093477702055, 0.0003859005258180428,
                            7.4611207452221e-5, 0.000149222414904442,
                            0.000298444829808884],
                        linf=[0.0013324014301672943, 0.0013324014301669181,
                            0.002684449324758791, 0.0016236816790307085,
                            0.0019172373117153363, 0.0019172373117148922,
                            0.002664932274107224, 0.0011872396664042962,
                            0.0002855492944235094, 0.0005710985888470188,
                            0.0011421971776940376])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_mhdmulti_rotor.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_mhdmulti_rotor.jl"),
                        l2=[0.6574459522153201, 0.6620356383023878, 0.0,
                            0.6888912144519942,
                            0.04882939911229928, 0.08366520368549821, 0.0,
                            0.0021850987869278136,
                            0.15909935226497424, 0.07954967613248712],
                        linf=[9.363623690550916, 9.178740037372911, 0.0,
                            10.611054196904469,
                            0.6628358023789442, 1.419291349928299, 0.0,
                            0.0988733910381692,
                            3.3287658922602334, 1.6643829461301167],
                        tspan=(0.0, 0.01))
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
