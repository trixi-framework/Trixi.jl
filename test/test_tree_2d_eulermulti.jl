@testsnippet TreeMesh2DEulerMulti begin
    EXAMPLES_DIR = joinpath(examples_dir(), "tree_2d_dgsem")
end

@testitem "TreeMesh2D EulerMulti: Testing entropy2cons and cons2entropy" setup=[Setup] tags=[:tree_part2] begin
    using ForwardDiff
    using Trixi: Trixi, CompressibleEulerMulticomponentEquations2D, cons2entropy,
                 entropy2cons, SVector
    gammas = (1.1546412974182538, 1.1171560258914812, 1.097107661471476,
              1.0587601652669245, 1.6209889683979308, 1.6732209755396386,
              1.2954303574165822)
    gas_constants = (5.969461071171914, 3.6660802003290183, 6.639008614675539,
                     8.116604827140456, 6.190706056680031, 1.6795013743693712,
                     2.197737590916966)
    equations = CompressibleEulerMulticomponentEquations2D(gammas = SVector{length(gammas)}(gammas...),
                                                           gas_constants = SVector{length(gas_constants)}(gas_constants...))
    u = [-1.7433292819144075, 0.8844413258376495, 0.6050737175812364,
        0.8261998359817043, 1.0801186290896465, 0.505654488367698,
        0.6364415555805734, 0.851669392285058, 0.31219606420306223,
        1.0930477805612038]
    w = cons2entropy(u, equations)
    # test that the entropy variables match the gradients of the total entropy
    @test w ≈ ForwardDiff.gradient(u -> Trixi.total_entropy(u, equations), u)
    # test that `entropy2cons` is the inverse of `cons2entropy`
    @test entropy2cons(w, equations) ≈ u
end

@testitem "TreeMesh2D EulerMulti: entropy potential" setup=[Setup] tags=[:tree_part2] begin
    using LinearAlgebra: dot, norm
    gammas = (1.4, 1.6)
    gas_constants = (1.0, 2.0)
    equations = CompressibleEulerMulticomponentEquations2D(; gammas, gas_constants)

    q_ll = SVector(-0.1, 0.2, 1.0, 1.1, 0.9)
    q_rr = SVector(0.2, -0.3, 2.0, 1.9, 2.1)
    u_ll, u_rr = prim2cons.((q_ll, q_rr), equations)

    # check that `flux_chandrashekar` is entropy conservative
    w_ll, w_rr = cons2entropy.((u_ll, u_rr), equations)
    jump_entropy_potential_x = entropy_potential(u_rr, 1, equations) -
                               entropy_potential(u_ll, 1, equations)
    jump_entropy_potential_y = entropy_potential(u_rr, 2, equations) -
                               entropy_potential(u_ll, 2, equations)
    @test dot(w_rr - w_ll, flux_chandrashekar(u_ll, u_rr, 1, equations)) ≈
          jump_entropy_potential_x
    @test dot(w_rr - w_ll, flux_chandrashekar(u_ll, u_rr, 2, equations)) ≈
          jump_entropy_potential_y

    normal_directions = [SVector(1.0, 0.0), SVector(0.0, 1.0)]
    for (orientation, normal_direction) in enumerate(normal_directions)
        for u in (u_ll, u_rr)
            @test entropy_potential(u, normal_direction, equations) ≈
                  entropy_potential(u, orientation, equations)
        end
    end

    p_ll = pressure(u_ll, equations)
    p_rr = pressure(u_rr, equations)
    p_avg = 0.5f0 * (p_ll + p_rr)
    vel_ll = velocity(u_ll, equations)
    vel_rr = velocity(u_rr, equations)
    vel_avg = 0.5f0 * (vel_ll + vel_rr)

    # check that `flux_srinivasan_nadarajah` is entropy conservative in x direction
    F_x = flux_srinivasan_nadarajah(u_ll, u_rr, 1, equations)
    tadmor_residual_x = dot(w_rr - w_ll, F_x) - jump_entropy_potential_x
    atol_ec_x = 100 * eps(Float64) * max(1, abs(jump_entropy_potential_x))
    @test abs(tadmor_residual_x) <= atol_ec_x

    # check Jameson KEP form for `flux_srinivasan_nadarajah` in x direction
    f_rho_sum_x = sum(@view F_x[4:end])
    kep_residual_x = @view(F_x[1:2]) - (f_rho_sum_x * vel_avg + p_avg * SVector(1.0, 0.0))
    atol_kep_x = 100 * eps(Float64) * max(1, norm(@view F_x[1:2]))
    @test norm(kep_residual_x) <= atol_kep_x

    # check that `flux_srinivasan_nadarajah` is entropy conservative in y direction
    F_y = flux_srinivasan_nadarajah(u_ll, u_rr, 2, equations)
    tadmor_residual_y = dot(w_rr - w_ll, F_y) - jump_entropy_potential_y
    atol_ec_y = 100 * eps(Float64) * max(1, abs(jump_entropy_potential_y))
    @test abs(tadmor_residual_y) <= atol_ec_y

    # check Jameson KEP form for `flux_srinivasan_nadarajah` in y direction
    f_rho_sum_y = sum(@view F_y[4:end])
    kep_residual_y = @view(F_y[1:2]) - (f_rho_sum_y * vel_avg + p_avg * SVector(0.0, 1.0))
    atol_kep_y = 100 * eps(Float64) * max(1, norm(@view F_y[1:2]))
    @test norm(kep_residual_y) <= atol_kep_y

    normal_direction_oblique = SVector(0.5, -1.0)
    jump_entropy_potential_oblique = entropy_potential(u_rr, normal_direction_oblique,
                                                       equations) -
                                     entropy_potential(u_ll, normal_direction_oblique,
                                                       equations)
    F_oblique = flux_srinivasan_nadarajah(u_ll, u_rr, normal_direction_oblique, equations)
    tadmor_residual_oblique = dot(w_rr - w_ll, F_oblique) - jump_entropy_potential_oblique
    atol_ec_oblique = 100 * eps(Float64) * max(1, abs(jump_entropy_potential_oblique))
    @test abs(tadmor_residual_oblique) <= atol_ec_oblique

    f_rho_sum_oblique = sum(@view F_oblique[4:end])
    kep_residual_oblique = @view(F_oblique[1:2]) -
                           (f_rho_sum_oblique * vel_avg + p_avg * normal_direction_oblique)
    atol_kep_oblique = 100 * eps(Float64) * max(1, norm(@view F_oblique[1:2]))
    @test norm(kep_residual_oblique) <= atol_kep_oblique
end

# NOTE: Some of the L2/Linf errors are comparably large. This is due to the fact that some of the
#       simulations are set up with dimensional states. For example, the reference pressure in SI
#       units is 101325 Pa, i.e., pressure has values of O(10^5)

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_shock_bubble.jl" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_shock_bubble.jl"),
                        l2=[
                            73.78467629094177,
                            0.9174752929795251,
                            57942.83587826468,
                            0.1828847253029943,
                            0.011127037850925347
                        ],
                        linf=[
                            196.81051991521073,
                            7.8456811648529605,
                            158891.88930113698,
                            0.811379581519794,
                            0.08011973559187913
                        ],
                        tspan=(0.0, 0.001))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_shock_bubble_shockcapturing_subcell_positivity.jl" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    rm(joinpath("out", "deviations.txt"), force = true)
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_eulermulti_shock_bubble_shockcapturing_subcell_positivity.jl"),
                        l2=[
                            81.52845664909304,
                            2.5455678559421346,
                            63229.190712645846,
                            0.19929478404550321,
                            0.011068604228443425
                        ],
                        linf=[
                            249.21708417382013,
                            40.33299887640794,
                            174205.0118831558,
                            0.6881458768113586,
                            0.11274401158173972
                        ],
                        initial_refinement_level=3,
                        tspan=(0.0, 0.001),
                        save_errors=true)
    lines = readlines(joinpath("out", "deviations.txt"))
    @test lines[1] == "# iter, simu_time, rho1_min, rho2_min"
    # Runs 15 time steps.
    @test startswith(lines[end], "15")

    limiter = semi.solver.volume_integral.limiter
    deviations = collect(values(limiter.cache.idp_bounds_delta_global))
    @test all(isfinite, deviations)
    @test maximum(deviations) <= 1.0e-13

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_shock_bubble_shockcapturing_subcell_minmax.jl" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR,
                                 "elixir_eulermulti_shock_bubble_shockcapturing_subcell_minmax.jl"),
                        l2=[
                            73.4058621244428,
                            1.506203467166868,
                            57401.37958800992,
                            0.17875593609409407,
                            0.01008524475706747
                        ],
                        linf=[
                            213.59140539012145,
                            24.575141945974114,
                            152498.2167599955,
                            0.5911106516013177,
                            0.09936092770249982
                        ],
                        initial_refinement_level=3,
                        tspan=(0.0, 0.001))
    limiter = semi.solver.volume_integral.limiter
    deviations = collect(values(limiter.cache.idp_bounds_delta_global))
    @test all(isfinite, deviations)
    @test maximum(deviations) <= 1.0e-13

    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    # Larger values for allowed allocations due to usage of custom
    # integrator which are not *recorded* for the methods from
    # OrdinaryDiffEq.jl
    # Corresponding issue: https://github.com/trixi-framework/Trixi.jl/issues/1877
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 15000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_ec.jl" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
                        l2=[
                            0.050182236154087095,
                            0.050189894464434635,
                            0.2258715597305131,
                            0.06175171559771687
                        ],
                        linf=[
                            0.3108124923284472,
                            0.3107380389947733,
                            1.054035804988521,
                            0.29347582879608936
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_ec.jl with flux_srinivasan_nadarajah" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
                        l2=[
                            0.05018223615408725,
                            0.05018989446443491,
                            0.22587155973051345,
                            0.06175171559771698
                        ],
                        linf=[
                            0.31081249232844776,
                            0.31073803899477576,
                            1.0540358049885197,
                            0.293475828796089
                        ],
                        surface_flux=flux_srinivasan_nadarajah,
                        volume_flux=flux_srinivasan_nadarajah)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_es.jl" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_es.jl"),
                        l2=[
                            0.0496546258404055,
                            0.04965550099933263,
                            0.22425206549856372,
                            0.004087155041747821,
                            0.008174310083495642,
                            0.016348620166991283,
                            0.032697240333982566
                        ],
                        linf=[
                            0.2488251110766228,
                            0.24832493304479406,
                            0.9310354690058298,
                            0.017452870465607374,
                            0.03490574093121475,
                            0.0698114818624295,
                            0.139622963724859
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_convergence_ec.jl" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_ec.jl"),
                        l2=[
                            0.00012290225488326508,
                            0.00012290225488321876,
                            0.00018867397906337653,
                            4.8542321753649044e-5,
                            9.708464350729809e-5
                        ],
                        linf=[
                            0.0006722819239133315,
                            0.0006722819239128874,
                            0.0012662292789555885,
                            0.0002843844182700561,
                            0.0005687688365401122
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_convergence_es.jl" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
                        l2=[
                            2.2661773867001696e-6,
                            2.266177386666318e-6,
                            6.593514692980009e-6,
                            8.836308667348217e-7,
                            1.7672617334696433e-6
                        ],
                        linf=[
                            1.4713170997993075e-5,
                            1.4713170997104896e-5,
                            5.115618808515521e-5,
                            5.3639516094383666e-6,
                            1.0727903218876733e-5
                        ])
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_convergence_es.jl with flux_chandrashekar" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_convergence_es.jl"),
                        l2=[
                            1.8621737639352465e-6,
                            1.862173764098385e-6,
                            5.942585713809631e-6,
                            6.216263279534722e-7,
                            1.2432526559069443e-6
                        ],
                        linf=[
                            1.6235495582606063e-5,
                            1.6235495576388814e-5,
                            5.854523678827661e-5,
                            5.790274858807898e-6,
                            1.1580549717615796e-5
                        ],
                        volume_flux=flux_chandrashekar)
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end

@testitem "TreeMesh2D EulerMulti: elixir_eulermulti_ec.jl with boundary_condition_slip_wall" setup=[
    Setup,
    TreeMesh2DEulerMulti
] tags=[:tree_part2] begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulermulti_ec.jl"),
                        l2=[
                            0.005884923780995506,
                            0.005815148890905981,
                            0.02343885021110439,
                            0.00625410669701958
                        ],
                        linf=[
                            0.24483047700349253,
                            0.13364458078315494,
                            0.3846939874019486,
                            0.1024647566986494
                        ],
                        periodicity=false,
                        boundary_conditions=boundary_condition_slip_wall,
                        tspan=(0.0, 0.001))
    # Ensure that we do not have excessive memory allocations
    # (e.g., from type instabilities)
    @test_allocations(Trixi.rhs_hyperbolic!, semi, sol, 1000)
end
