module TestExamples1DEuler

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = pkgdir(Trixi, "examples", "tree_1d_dgsem")

@testset "Compressible Euler" begin
#! format: noindent

@trixi_testset "elixir_euler_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms.jl"),
                        l2=[
                            2.2527950196212703e-8,
                            1.8187357193835156e-8,
                            7.705669939973104e-8
                        ],
                        linf=[
                            1.6205433861493646e-7,
                            1.465427772462391e-7,
                            5.372255111879554e-7
                        ],
                        # With the default `maxiters = 1` in coverage tests,
                        # there would be no time series to check against.
                        coverage_override=(maxiters = 20,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
    # Extra test to make sure the "TimeSeriesCallback" made correct data.
    # Extracts data at all points from the first step of the time series and compares it to the 
    # exact solution and an interpolated reference solution
    point_data = [getindex(time_series.affect!.point_data[i], 1:3) for i in 1:3]
    exact_data = [initial_condition_convergence_test(time_series.affect!.point_coordinates[i],
                                                     time_series.affect!.time[1],
                                                     equations) for i in 1:3]
    ref_data = [[1.968279088772251, 1.9682791565395945, 3.874122958278797],
        [2.0654816955822017, 2.0654817326611883, 4.26621471136323],
        [2.0317209235018936, 2.0317209516429506, 4.127889808862571]]
    @test point_data≈exact_data atol=1e-6
    @test point_data ≈ ref_data
end

@trixi_testset "elixir_euler_convergence_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_convergence_pure_fv.jl"),
                        l2=[
                            0.019355699748523896,
                            0.022326984561234497,
                            0.02523665947241734
                        ],
                        linf=[
                            0.02895961127645519,
                            0.03293442484199227,
                            0.04246098278632804
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

@trixi_testset "elixir_euler_density_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
                        l2=[
                            0.0011482554820217855,
                            0.00011482554830323462,
                            5.741277429325267e-6
                        ],
                        linf=[
                            0.004090978306812376,
                            0.0004090978313582294,
                            2.045489210189544e-5
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

@trixi_testset "elixir_euler_density_wave.jl with initial_condition_constant" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_density_wave.jl"),
                        l2=[
                            7.71293052584723e-16,
                            1.9712947511091717e-14,
                            7.50672833504266e-15
                        ],
                        linf=[
                            3.774758283725532e-15,
                            6.733502644351574e-14,
                            2.4868995751603507e-14
                        ],
                        initial_condition=initial_condition_constant)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_source_terms_nonperiodic.jl"),
                        l2=[
                            3.8099996914101204e-6,
                            1.6745575717106341e-6,
                            7.732189531480852e-6
                        ],
                        linf=[
                            1.2971473393186272e-5,
                            9.270328934274374e-6,
                            3.092514399671842e-5
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
                            0.11821957357197649,
                            0.15330089521538678,
                            0.4417674632047301
                        ],
                        linf=[
                            0.24280567569982958,
                            0.29130548795961936,
                            0.8847009003152442
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

@trixi_testset "elixir_euler_ec.jl with flux_kennedy_gruber" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.07803455838661963,
                            0.10032577312032283,
                            0.29228156303827935
                        ],
                        linf=[
                            0.2549869853794955,
                            0.3376472164661263,
                            0.9650477546553962
                        ],
                        maxiters=10,
                        surface_flux=flux_kennedy_gruber,
                        volume_flux=flux_kennedy_gruber)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_ec.jl with flux_shima_etal" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.07800654460172655,
                            0.10030365573277883,
                            0.2921481199111959
                        ],
                        linf=[
                            0.25408579350400395,
                            0.3388657679031271,
                            0.9776486386921928
                        ],
                        maxiters=10,
                        surface_flux=flux_shima_etal,
                        volume_flux=flux_shima_etal)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_ec.jl with flux_chandrashekar" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[
                            0.07801923089205756,
                            0.10039557434912669,
                            0.2922210399923278
                        ],
                        linf=[
                            0.2576521982607225,
                            0.3409717926625057,
                            0.9772961936567048
                        ],
                        maxiters=10,
                        surface_flux=flux_chandrashekar,
                        volume_flux=flux_chandrashekar)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_ec.jl with flux_hll" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_ec.jl"),
                        l2=[0.07855251823583848, 0.10213903748267686, 0.293985892532479],
                        linf=[
                            0.192621556068018,
                            0.25184744005299536,
                            0.7264977555504792
                        ],
                        maxiters=10,
                        surface_flux=flux_hll,
                        volume_flux=flux_ranocha)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_shockcapturing.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing.jl"),
                        l2=[
                            0.11606096465319675,
                            0.15028768943458806,
                            0.4328230323046703
                        ],
                        linf=[
                            0.18031710091067965,
                            0.2351582421501841,
                            0.6776805692092567
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

@trixi_testset "elixir_euler_sedov_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
                        l2=[1.250005061244617, 0.06878411345533507, 0.9264328311018613],
                        linf=[
                            2.9766770877037168,
                            0.16838100902295852,
                            2.6655773445485798
                        ],
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave.jl (HLLE)" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
                        l2=[0.6442208390304879, 0.508817280068289, 0.9482809853033687],
                        linf=[3.007059066482486, 2.4678899558345506, 2.3952311739389787],
                        tspan=(0.0, 0.5),
                        surface_flux=flux_hlle)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave_pure_fv.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_sedov_blast_wave_pure_fv.jl"),
                        l2=[1.0735456065491455, 0.07131078703089379, 0.9205739468590453],
                        linf=[
                            3.4296365168219216,
                            0.17635583964559245,
                            2.6574584326179505
                        ],
                        # Let this test run longer to cover some lines in flux_hllc
                        coverage_override=(maxiters = 10^5, tspan = (0.0, 0.1)))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave.jl with pressure" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
                        l2=[1.297525985166995, 0.07964929522694145, 0.9269991156246368],
                        linf=[
                            3.1773015255764427,
                            0.21331831536493773,
                            2.6650170188241047
                        ],
                        shock_indicator_variable=pressure,
                        cfl=0.2,
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_sedov_blast_wave.jl with density" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov_blast_wave.jl"),
                        l2=[1.2798798835860528, 0.07103461242058921, 0.9273792517187003],
                        linf=[
                            3.1087017048015824,
                            0.17734706962928956,
                            2.666689753470263
                        ],
                        shock_indicator_variable=density,
                        cfl=0.2,
                        coverage_override=(maxiters = 6,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_positivity.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_positivity.jl"),
                        l2=[1.6493820253458906, 0.19793887460986834, 0.9783506076125921],
                        linf=[4.71751203912051, 0.5272411022735763, 2.7426163947635844],
                        coverage_override=(maxiters = 3,))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blast_wave.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_blast_wave.jl"),
                        l2=[0.21934822867340323, 0.28131919126002686, 0.554361702716662],
                        linf=[
                            1.5180897390290355,
                            1.3967085956620369,
                            2.0663825294019595
                        ],
                        maxiters=30)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "elixir_euler_blast_wave_entropy_bounded.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_blast_wave_entropy_bounded.jl"),
                        l2=[0.9689207881108007, 0.1617708899929322, 1.3847895715669456],
                        linf=[2.95591859210077, 0.3135723412205586, 2.3871554358655365])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    let
        t = sol.t[end]
        u_ode = sol.u[end]
        du_ode = similar(u_ode)
        @test (@allocated Trixi.rhs!(du_ode, u_ode, semi, t)) < 1000
    end
end

@trixi_testset "test_quasi_1D_entropy" begin
    a = 0.9
    u_1D = SVector(1.1, 0.2, 2.1)
    u_quasi_1D = SVector(a * 1.1, a * 0.2, a * 2.1, a)
    @test entropy(u_quasi_1D, CompressibleEulerEquationsQuasi1D(1.4)) ≈
          a * entropy(u_1D, CompressibleEulerEquations1D(1.4))
end

@trixi_testset "elixir_euler_quasi_1d_source_terms.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_quasi_1d_source_terms.jl"),
                        l2=[
                            3.876288369618363e-7,
                            2.2247043122302947e-7,
                            2.964004224572679e-7,
                            5.2716983399807875e-8
                        ],
                        linf=[
                            2.3925118561862746e-6,
                            1.3603693522767912e-6,
                            1.821888865105592e-6,
                            1.1166012159335992e-7
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

@trixi_testset "elixir_euler_quasi_1d_discontinuous.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_euler_quasi_1d_discontinuous.jl"),
                        l2=[
                            0.045510421156346015,
                            0.036750584788912195,
                            0.2468985959132176,
                            0.03684494180829024
                        ],
                        linf=[
                            0.3313374853025697,
                            0.11621933362158643,
                            1.827403013568638,
                            0.28045939999015723
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

@trixi_testset "elixir_euler_quasi_1d_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_quasi_1d_ec.jl"),
                        l2=[
                            0.08889113985713998,
                            0.16199235348889673,
                            0.40316524365054346,
                            2.9602775074723667e-16
                        ],
                        linf=[
                            0.28891355898284043,
                            0.3752709888964313,
                            0.84477102402413,
                            8.881784197001252e-16
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
end

end # module
