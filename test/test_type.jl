@testsnippet TypeStability begin
    using ForwardDiff
end

@testitem "Type stability: mean values" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT1 in (Float32, Float64), RealT2 in (Float32, Float64)
        RealT = promote_type(RealT1, RealT2)
        @test typeof(@inferred Trixi.ln_mean(RealT1(1), RealT2(2))) == RealT
        @test typeof(@inferred Trixi.inv_ln_mean(RealT1(1), RealT2(2))) == RealT
        for RealT3 in (Float32, Float64)
            RealT = promote_type(RealT1, RealT2, RealT3)
            @test typeof(@inferred Trixi.stolarsky_mean(RealT1(1), RealT2(2),
                                                        RealT3(3))) ==
                  RealT
        end
    end
end

@testitem "Type stability: TreeMesh & SerialTree type consistence" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        coordinates_min = -convert(RealT, 1)
        coordinates_max = convert(RealT, 1)

        mesh = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 6,
                        n_cells_max = 30_000,
                        RealT = RealT, periodicity = true)

        @test typeof(@inferred Trixi.total_volume(mesh)) == RealT

        coordinates_min = (-convert(RealT, 42), -convert(RealT, 42))
        coordinates_max = (convert(RealT, 42), convert(RealT, 42))

        mesh = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 5,
                        n_cells_max = 30_000,
                        RealT = RealT, periodicity = true)

        @test typeof(@inferred Trixi.total_volume(mesh)) == RealT

        coordinates_min = (-convert(RealT, pi), -convert(RealT, pi),
                           -convert(RealT, pi))
        coordinates_max = (convert(RealT, pi), convert(RealT, pi),
                           convert(RealT, pi))

        mesh = TreeMesh(coordinates_min, coordinates_max,
                        initial_refinement_level = 4,
                        n_cells_max = 30_000,
                        RealT = RealT, periodicity = true)

        @test typeof(@inferred Trixi.total_volume(mesh)) == RealT
    end
end

@testitem "Type stability: Acoustic Perturbation 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        v_mean_global = (zero(RealT), zero(RealT))
        c_mean_global = one(RealT)
        rho_mean_global = one(RealT)
        equations = @inferred AcousticPerturbationEquations2D(v_mean_global,
                                                              c_mean_global,
                                                              rho_mean_global)

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT),
                                            one(RealT),
                                            one(RealT),
                                            one(RealT), one(RealT))
        orientations = [1, 2]
        directions = [1, 2, 3, 4]
        normal_direction = SVector(one(RealT), zero(RealT))

        surface_flux_function = flux_lax_friedrichs
        dissipation = DissipationLocalLaxFriedrichs()

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_gauss(x, t, equations)) == RealT

        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred boundary_condition_wall(u_inner, orientation,
                                                               direction, x, t,
                                                               surface_flux_function,
                                                               equations)) == RealT
            end
        end
        @test eltype(@inferred boundary_condition_slip_wall(u_inner,
                                                            normal_direction, x, t,
                                                            surface_flux_function,
                                                            equations)) ==
              RealT

        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) ==
              RealT
        @test eltype(@inferred dissipation(u_ll, u_rr, normal_direction, equations)) ==
              RealT

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred dissipation(u_ll, u_rr, orientation, equations)) ==
                  RealT
        end

        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa AcousticPerturbationEquations2D{Float32}
        @test eltype(adapted.v_mean_global) == Float32
        @test typeof(adapted.c_mean_global) == Float32
    end
end

@testitem "Type stability: Compressible Euler 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        # set gamma = 2 for the coupling convergence test
        equations = @inferred CompressibleEulerEquations1D(RealT(2))

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = cons = SVector(one(RealT), one(RealT), one(RealT))
        orientation = 1
        directions = [1, 2]

        surface_flux_function = flux_lax_friedrichs

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_density_wave(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_eoc_test_coupled_euler_gravity(x, t,
                                                                                equations)) ==
              RealT

        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT

        for direction in directions
            @test eltype(@inferred boundary_condition_slip_wall(u_inner, orientation,
                                                                direction,
                                                                x, t,
                                                                surface_flux_function,
                                                                equations)) ==
                  RealT
        end

        @test eltype(@inferred flux(u, orientation, equations)) == RealT
        @test eltype(@inferred flux_shima_etal(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred flux_kennedy_gruber(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred flux_hllc(u_ll, u_rr, orientation, equations)) == RealT
        @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                  equations)) ==
              RealT
        @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
              RealT

        @test eltype(eltype(@inferred splitting_steger_warming(u, orientation,
                                                               equations))) ==
              RealT
        @test eltype(eltype(@inferred splitting_vanleer_haenel(u, orientation,
                                                               equations))) ==
              RealT
        @test eltype(eltype(@inferred splitting_coirier_vanleer(u, orientation,
                                                                equations))) ==
              RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                      equations)) ==
              RealT

        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test eltype(@inferred entropy2cons(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred velocity(u, equations)) == RealT
        @test typeof(@inferred velocity(u, orientation, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT
        @test typeof(@inferred entropy(cons, equations)) == RealT
        @test typeof(@inferred energy_internal(cons, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa CompressibleEulerEquations1D{Float32}
        @test typeof(adapted.gamma) == Float32
        @test typeof(adapted.inv_gamma_minus_one) == Float32
    end
end

@testitem "Type stability: NonIdeal Compressible Euler 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations_ideal_gas = @inferred NonIdealCompressibleEulerEquations1D(IdealGas(RealT(2)))
        a, b, gamma, R = RealT.((0.0, 0.0, 1.4, 287))
        equations_vdw = @inferred NonIdealCompressibleEulerEquations1D(VanDerWaals(; a,
                                                                                   b,
                                                                                   gamma,
                                                                                   R))
        equations_helmholtz_ideal_gas = @inferred NonIdealCompressibleEulerEquations1D(HelmholtzIdealGas(RealT(2)))
        for equations in (equations_ideal_gas, equations_vdw,
                          equations_helmholtz_ideal_gas)
            x = SVector(zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = u_inner = cons = SVector(one(RealT), one(RealT),
                                                       one(RealT))
            orientation = 1
            direction = 1

            surface_flux_function = flux_lax_friedrichs

            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_terashima_etal(u, u, orientation, equations)) ==
                  RealT
            @test eltype(@inferred flux_central_terashima_etal(u, u, orientation,
                                                               equations)) == RealT
            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT

            q = cons2thermo(u, equations)
            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred cons2thermo(u, equations)) == RealT
            @test eltype(@inferred thermo2cons(q, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            # TODO: if entropy2cons is implemented, add a test

            @test typeof(@inferred density(u, equations)) == RealT
            @test typeof(@inferred velocity(u, equations)) == RealT
            @test typeof(@inferred velocity(u, orientation, equations)) == RealT
            @test typeof(@inferred pressure(u, equations)) == RealT
            @test typeof(@inferred density_pressure(u, equations)) == RealT
            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
        end

        # EoS adapt tests
        adapted_ig = @inferred Trixi.trixi_adapt(Array, Float32,
                                                 equations_ideal_gas.equation_of_state)
        @test typeof(adapted_ig.gamma) == Float32
        @test typeof(adapted_ig.cv) == Float32
        adapted_vdw = @inferred Trixi.trixi_adapt(Array, Float32,
                                                  equations_vdw.equation_of_state)
        @test typeof(adapted_vdw.a) == Float32
        @test typeof(adapted_vdw.cv) == Float32
        eos_pr = Trixi.PengRobinson(RealT(0.5), RealT(0.1), RealT(0.7), RealT(0.3),
                                    RealT(300), RealT(8.314))
        adapted_pr = @inferred Trixi.trixi_adapt(Array, Float32, eos_pr)
        @test typeof(adapted_pr.R) == Float32
        @test typeof(adapted_pr.inv2sqrt2b) == Float32
        adapted_heim = @inferred Trixi.trixi_adapt(Array, Float32,
                                                   equations_helmholtz_ideal_gas.equation_of_state)
        @test typeof(adapted_heim.gamma) == Float32
        @test typeof(adapted_heim.R) == Float32

        # Wrapper adapt tests
        adapted_neq = @inferred Trixi.trixi_adapt(Array, Float32, equations_ideal_gas)
        @test adapted_neq isa NonIdealCompressibleEulerEquations1D
        @test typeof(adapted_neq.equation_of_state.gamma) == Float32
    end
end

@testitem "Type stability: NonIdeal Compressible Euler 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations_ideal_gas = @inferred NonIdealCompressibleEulerEquations2D(IdealGas(RealT(2)))
        a, b, gamma, R = RealT.((0.0, 0.0, 1.4, 287))
        equations_vdw = @inferred NonIdealCompressibleEulerEquations1D(VanDerWaals(; a,
                                                                                   b,
                                                                                   gamma,
                                                                                   R))

        for equations in (equations_ideal_gas, equations_vdw)
            x = SVector(zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = u_inner = cons = SVector(one(RealT), one(RealT),
                                                       one(RealT), one(RealT))
            orientation = 1
            direction = 1

            surface_flux_function = flux_lax_friedrichs

            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_terashima_etal(u, u, orientation, equations)) ==
                  RealT
            @test eltype(@inferred flux_central_terashima_etal(u, u, orientation,
                                                               equations)) == RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT

            q = cons2thermo(u, equations)
            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred cons2thermo(u, equations)) == RealT
            @test eltype(@inferred thermo2cons(q, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            # TODO: if entropy2cons is implemented, add a test

            @test typeof(@inferred Trixi.density(u, equations)) == RealT
            @test eltype(@inferred velocity(u, equations)) == RealT
            @test typeof(@inferred velocity(u, orientation, equations)) == RealT
            @test typeof(@inferred pressure(u, equations)) == RealT
            @test typeof(@inferred density_pressure(u, equations)) == RealT
            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
        end

        adapted_neq2d = @inferred Trixi.trixi_adapt(Array, Float32, equations_ideal_gas)
        @test adapted_neq2d isa NonIdealCompressibleEulerEquations2D
        @test typeof(adapted_neq2d.equation_of_state.gamma) == Float32
    end
end

@testitem "Type stability: Compressible Euler 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        # set gamma = 2 for the coupling convergence test
        equations = @inferred CompressibleEulerEquations2D(RealT(2))

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = cons = SVector(one(RealT), one(RealT), one(RealT),
                                                   one(RealT))
        orientations = [1, 2]
        directions = [1, 2, 3, 4]
        normal_direction = SVector(one(RealT), zero(RealT))

        surface_flux_function = flux_lax_friedrichs
        flux_lmars = FluxLMARS(340)

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_density_wave(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_eoc_test_coupled_euler_gravity(x, t,
                                                                                equations)) ==
              RealT

        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_eoc_test_coupled_euler_gravity(u, x, t,
                                                                           equations)) ==
              RealT
        @test eltype(@inferred source_terms_eoc_test_euler(u, x, t, equations)) ==
              RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred boundary_condition_slip_wall(u_inner,
                                                                    orientation,
                                                                    direction, x, t,
                                                                    surface_flux_function,
                                                                    equations)) == RealT
            end
        end

        @test eltype(@inferred velocity(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_shima_etal(u_ll, u_rr, normal_direction, equations)) ==
              RealT
        @test eltype(@inferred flux_kennedy_gruber(u_ll, u_rr, normal_direction,
                                                   equations)) ==
              RealT
        @test eltype(@inferred flux_lmars(u_ll, u_rr, normal_direction, equations)) ==
              RealT
        @test eltype(@inferred flux_hllc(u_ll, u_rr, normal_direction, equations)) ==
              RealT
        @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, normal_direction,
                                                  equations)) ==
              RealT
        @test eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                            equations)) == RealT

        @test eltype(eltype(@inferred splitting_drikakis_tsangaris(u, normal_direction,
                                                                   equations))) == RealT
        @test eltype(eltype(@inferred splitting_vanleer_haenel(u, normal_direction,
                                                               equations))) ==
              RealT
        @test eltype(eltype(@inferred splitting_lax_friedrichs(u, normal_direction,
                                                               equations))) ==
              RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                   equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, normal_direction,
                                                      equations)) ==
              RealT

        for orientation in orientations
            @test eltype(@inferred velocity(u, orientation, equations)) == RealT
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_shima_etal(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred flux_kennedy_gruber(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred flux_lmars(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred flux_hllc(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                      equations)) ==
                  RealT
            @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                  RealT

            @test eltype(eltype(@inferred splitting_steger_warming(u, orientation,
                                                                   equations))) ==
                  RealT
            @test eltype(eltype(@inferred splitting_drikakis_tsangaris(u, orientation,
                                                                       equations))) ==
                  RealT
            @test eltype(eltype(@inferred splitting_vanleer_haenel(u, orientation,
                                                                   equations))) ==
                  RealT
            @test eltype(eltype(@inferred splitting_lax_friedrichs(u, orientation,
                                                                   equations))) ==
                  RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                          equations)) ==
                  RealT
        end

        @test eltype(@inferred velocity(u, equations)) == RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test eltype(@inferred entropy2cons(u, equations)) == RealT
        @test eltype(@inferred Trixi.cons2entropy_guermond_etal(u, equations)) == RealT
        @test typeof(@inferred entropy_guermond_etal(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT
        @test typeof(@inferred entropy(cons, equations)) == RealT
        @test typeof(@inferred entropy_math(cons, equations)) == RealT
        @test typeof(@inferred entropy_thermodynamic(cons, equations)) == RealT
        @test typeof(@inferred energy_internal(cons, equations)) == RealT

        @test eltype(@inferred Trixi.gradient_conservative(pressure, u, equations)) ==
              RealT
        @test eltype(@inferred Trixi.gradient_conservative(entropy_math, u,
                                                           equations)) == RealT
        @test eltype(@inferred Trixi.gradient_conservative(entropy_guermond_etal,
                                                           u,
                                                           equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa CompressibleEulerEquations2D{Float32}
        @test typeof(adapted.gamma) == Float32
        @test typeof(adapted.inv_gamma_minus_one) == Float32
    end
end

@testitem "Type stability: Compressible Euler 3D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        # set gamma = 2 for the coupling convergence test
        equations = @inferred CompressibleEulerEquations3D(RealT(2))

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = cons = SVector(one(RealT), one(RealT), one(RealT),
                                                   one(RealT), one(RealT))
        orientations = [1, 2, 3]
        directions = [1, 2, 3, 4, 5, 6]
        normal_direction = SVector(one(RealT), zero(RealT), zero(RealT))

        surface_flux_function = flux_lax_friedrichs
        flux_lmars = FluxLMARS(340)

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_eoc_test_coupled_euler_gravity(x, t,
                                                                                equations)) ==
              RealT

        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_eoc_test_coupled_euler_gravity(u, x, t,
                                                                           equations)) ==
              RealT
        @test eltype(@inferred source_terms_eoc_test_euler(u, x, t, equations)) == RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred boundary_condition_slip_wall(u_inner,
                                                                    orientation,
                                                                    direction, x, t,
                                                                    surface_flux_function,
                                                                    equations)) == RealT
            end
        end

        @test eltype(@inferred velocity(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_shima_etal(u_ll, u_rr, normal_direction, equations)) ==
              RealT
        @test eltype(@inferred flux_kennedy_gruber(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred flux_lmars(u_ll, u_rr, normal_direction, equations)) ==
              RealT
        @test eltype(@inferred flux_hllc(u_ll, u_rr, normal_direction, equations)) ==
              RealT
        @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, normal_direction,
                                                  equations)) == RealT
        @test eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                            equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, normal_direction,
                                                      equations)) == RealT

        for orientation in orientations
            @test eltype(@inferred velocity(u, orientation, equations)) == RealT
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_shima_etal(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred flux_kennedy_gruber(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred flux_lmars(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred flux_hllc(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                      equations)) == RealT
            @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                  RealT

            @test eltype(eltype(@inferred splitting_steger_warming(u, orientation,
                                                                   equations))) ==
                  RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                          equations)) == RealT
        end

        @test eltype(@inferred velocity(u, equations)) == RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test eltype(@inferred entropy2cons(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT
        @test typeof(@inferred entropy(cons, equations)) == RealT
        @test typeof(@inferred entropy_math(cons, equations)) == RealT
        @test typeof(@inferred entropy_thermodynamic(cons, equations)) == RealT
        @test typeof(@inferred energy_internal(cons, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa CompressibleEulerEquations3D{Float32}
        @test typeof(adapted.gamma) == Float32
        @test typeof(adapted.inv_gamma_minus_one) == Float32
    end
end

@testitem "Type stability: Compressible Euler Multicomponent 1D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        gammas = (RealT(1.4), RealT(1.4))
        gas_constants = (RealT(0.4), RealT(0.4))
        equations = @inferred CompressibleEulerMulticomponentEquations1D(gammas = gammas,
                                                                         gas_constants = gas_constants)

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = SVector(one(RealT), one(RealT), one(RealT), one(RealT))
        orientation = 1

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT

        @test eltype(@inferred velocity(u, equations)) == RealT
        @test eltype(@inferred flux(u, orientation, equations)) == RealT
        @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                  equations)) == RealT
        @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
              RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test eltype(@inferred entropy2cons(u, equations)) == RealT
        @test typeof(@inferred Trixi.total_entropy(u, equations)) == RealT
        @test typeof(@inferred Trixi.temperature(u, equations)) == RealT
        @test typeof(@inferred Trixi.totalgamma(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa CompressibleEulerMulticomponentEquations1D
        @test eltype(adapted.gammas) == Float32
        @test eltype(adapted.cv) == Float32
    end
end

@testitem "Type stability: Compressible Euler Multicomponent 2D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        gammas = (RealT(1.4), RealT(1.4))
        gas_constants = (RealT(0.4), RealT(0.4))
        equations = @inferred CompressibleEulerMulticomponentEquations2D(gammas = gammas,
                                                                         gas_constants = gas_constants)

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = SVector(one(RealT), one(RealT), one(RealT), one(RealT),
                                  one(RealT))
        orientations = [1, 2]
        normal_direction = SVector(one(RealT), zero(RealT))

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT

        @test eltype(@inferred velocity(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                            equations)) == RealT

        for orientation in orientations
            @test eltype(@inferred velocity(u, orientation, equations)) == RealT
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                      equations)) == RealT
            @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                  RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
        end

        @test eltype(@inferred velocity(u, equations)) == RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test eltype(@inferred entropy2cons(u, equations)) == RealT
        @test typeof(@inferred Trixi.total_entropy(u, equations)) == RealT
        @test typeof(@inferred Trixi.temperature(u, equations)) == RealT
        @test typeof(@inferred Trixi.totalgamma(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa CompressibleEulerMulticomponentEquations2D
        @test eltype(adapted.gammas) == Float32
        @test eltype(adapted.cv) == Float32
    end
end

@testitem "Type stability: Compressible Euler Quasi 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquationsQuasi1D(RealT(1.4))

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = SVector(one(RealT), one(RealT), one(RealT), one(RealT))
        orientation = 1
        normal_direction = normal_ll = normal_rr = SVector(one(RealT))

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT

        @test eltype(@inferred flux(u, orientation, equations)) == RealT
        @test eltype(@inferred flux_nonconservative_chan_etal(u_ll, u_rr, orientation,
                                                              equations)) == RealT
        @test eltype(@inferred flux_nonconservative_chan_etal(u_ll, u_rr,
                                                              normal_direction,
                                                              equations)) ==
              RealT
        @test eltype(@inferred flux_nonconservative_chan_etal(u_ll, u_rr, normal_ll,
                                                              normal_rr, equations)) ==
              RealT
        @test eltype(@inferred flux_chan_etal(u_ll, u_rr, orientation, equations)) ==
              RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa CompressibleEulerEquationsQuasi1D{Float32}
        @test typeof(adapted.gamma) == Float32
        @test typeof(adapted.inv_gamma_minus_one) == Float32
    end
end

@testitem "Type stability: Compressible Navier Stokes Diffusion 1D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations1D(RealT(1.4))
        prandtl_number = RealT(0.72)
        mu = RealT(0.01)
        equations_parabolic_primitive = @inferred CompressibleNavierStokesDiffusion1D(equations,
                                                                                      mu = mu,
                                                                                      Prandtl = prandtl_number,
                                                                                      gradient_variables = GradientVariablesPrimitive())
        equations_parabolic_entropy = @inferred CompressibleNavierStokesDiffusion1D(equations,
                                                                                    mu = mu,
                                                                                    Prandtl = prandtl_number,
                                                                                    gradient_variables = GradientVariablesEntropy())

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_inner = u_transformed = flux_inner = SVector(one(RealT), zero(RealT),
                                                           zero(RealT))
        orientation = 1
        directions = [1, 2]
        gradients = SVector(RealT(0.1), RealT(0.1), RealT(0.1))

        operator_gradient = Trixi.Gradient()
        operator_divergence = Trixi.Divergence()

        # For BC tests
        function initial_condition_navier_stokes_convergence_test(x, t, equations)
            RealT_local = eltype(x)
            A = 0.5f0
            c = 2

            pi_x = convert(RealT_local, pi) * x[1]
            pi_t = convert(RealT_local, pi) * t

            rho = c + A * cos(pi_x) * cos(pi_t)
            v1 = log(x[1] + 2) * (1 - exp(-A * (x[1] - 1))) * cos(pi_t)
            p = rho^2

            return prim2cons(SVector(rho, v1, p), equations)
        end

        for equations_parabolic in (equations_parabolic_primitive,
                                    equations_parabolic_entropy)
            @test eltype(@inferred flux(u, (gradients,), orientation,
                                        equations_parabolic)) ==
                  RealT

            @test eltype(@inferred cons2prim(u, equations_parabolic)) == RealT
            @test eltype(@inferred prim2cons(u, equations_parabolic)) == RealT
            @test eltype(@inferred cons2entropy(u, equations_parabolic)) == RealT
            @test eltype(@inferred entropy2cons(u, equations_parabolic)) == RealT
            @test typeof(@inferred Trixi.temperature(u, equations_parabolic)) == RealT
            if equations_parabolic.gradient_variables isa GradientVariablesEntropy
                w = cons2entropy(u, equations_parabolic)
                @test eltype(@inferred Trixi.entropy2velocity_temperature(w,
                                                                          equations_parabolic)) ==
                      RealT
            end

            @test eltype(@inferred Trixi.convert_transformed_to_velocity_temperature(u_transformed,
                                                                                     equations_parabolic)) ==
                  RealT
            @test eltype(@inferred Trixi.convert_derivative_to_primitive(u, gradients,
                                                                         equations_parabolic)) ==
                  RealT

            # For BC tests
            velocity_bc_left_right = NoSlip((x, t, equations) -> initial_condition_navier_stokes_convergence_test(x,
                                                                                                                  t,
                                                                                                                  equations)[2])
            heat_bc_left = Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                              t,
                                                                                                                              equations),
                                                                             equations_parabolic))
            heat_bc_right = Adiabatic((x, t, equations) -> oftype(t, 0))

            boundary_condition_left = BoundaryConditionNavierStokesWall(velocity_bc_left_right,
                                                                        heat_bc_left)
            boundary_condition_right = BoundaryConditionNavierStokesWall(velocity_bc_left_right,
                                                                         heat_bc_right)
            # BC tests
            for direction in directions
                @test eltype(@inferred boundary_condition_right(flux_inner, u_inner,
                                                                orientation, direction,
                                                                x,
                                                                t, operator_gradient,
                                                                equations_parabolic)) ==
                      RealT
                @test eltype(@inferred boundary_condition_right(flux_inner, u_inner,
                                                                orientation, direction,
                                                                x,
                                                                t, operator_divergence,
                                                                equations_parabolic)) ==
                      RealT
                @test eltype(@inferred boundary_condition_left(flux_inner, u_inner,
                                                               orientation, direction,
                                                               x,
                                                               t, operator_gradient,
                                                               equations_parabolic)) ==
                      RealT
                @test eltype(@inferred boundary_condition_left(flux_inner, u_inner,
                                                               orientation, direction,
                                                               x,
                                                               t, operator_divergence,
                                                               equations_parabolic)) ==
                      RealT
            end
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32,
                                              equations_parabolic_primitive)
        @test adapted isa CompressibleNavierStokesDiffusion1D
        @test typeof(adapted.mu) == Float32
        @test typeof(adapted.Pr) == Float32
        @test typeof(adapted.kappa) == Float32
        @test adapted.equations_hyperbolic isa CompressibleEulerEquations1D{Float32}
    end
end

@testitem "Type stability: Compressible Navier Stokes Diffusion 2D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations2D(RealT(1.4))
        prandtl_number = RealT(0.72)
        mu = RealT(0.01)
        equations_parabolic_primitive = @inferred CompressibleNavierStokesDiffusion2D(equations,
                                                                                      mu = mu,
                                                                                      Prandtl = prandtl_number,
                                                                                      gradient_variables = GradientVariablesPrimitive())
        equations_parabolic_entropy = @inferred CompressibleNavierStokesDiffusion2D(equations,
                                                                                    mu = mu,
                                                                                    Prandtl = prandtl_number,
                                                                                    gradient_variables = GradientVariablesEntropy())

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = w_inner = u_transformed = flux_inner = normal = SVector(one(RealT),
                                                                    zero(RealT),
                                                                    zero(RealT),
                                                                    zero(RealT))
        orientations = [1, 2]
        gradient = SVector(RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1))
        gradients = SVector(gradient, gradient)

        operator_gradient = Trixi.Gradient()
        operator_divergence = Trixi.Divergence()

        # For BC tests
        function initial_condition_navier_stokes_convergence_test(x, t, equations)
            RealT_local = eltype(x)
            A = 0.5f0
            c = 2

            pi_x = convert(RealT_local, pi) * x[1]
            pi_y = convert(RealT_local, pi) * x[2]
            pi_t = convert(RealT_local, pi) * t

            rho = c + A * sin(pi_x) * cos(pi_y) * cos(pi_t)
            v1 = sin(pi_x) * log(x[2] + 2) * (1 - exp(-A * (x[2] - 1))) * cos(pi_t)
            v2 = v1
            p = rho^2

            return prim2cons(SVector(rho, v1, v2, p), equations)
        end

        for equations_parabolic in (equations_parabolic_primitive,
                                    equations_parabolic_entropy)
            for orientation in orientations
                @test eltype(@inferred flux(u, gradients, orientation,
                                            equations_parabolic)) == RealT
            end

            @test eltype(@inferred cons2prim(u, equations_parabolic)) == RealT
            @test eltype(@inferred prim2cons(u, equations_parabolic)) == RealT
            @test eltype(@inferred cons2entropy(u, equations_parabolic)) == RealT
            @test eltype(@inferred entropy2cons(u, equations_parabolic)) == RealT
            @test typeof(@inferred Trixi.temperature(u, equations_parabolic)) == RealT
            if equations_parabolic.gradient_variables isa GradientVariablesEntropy
                w = cons2entropy(u, equations_parabolic)
                @test eltype(@inferred Trixi.entropy2velocity_temperature(w,
                                                                          equations_parabolic)) ==
                      RealT
            end
            @test typeof(@inferred Trixi.enstrophy(u, gradients, equations_parabolic)) ==
                  RealT
            @test typeof(@inferred Trixi.vorticity(u, gradients, equations_parabolic)) ==
                  RealT

            @test eltype(@inferred Trixi.convert_transformed_to_velocity_temperature(u_transformed,
                                                                                     equations_parabolic)) ==
                  RealT
            @test eltype(@inferred Trixi.convert_derivative_to_primitive(u, gradient,
                                                                         equations_parabolic)) ==
                  RealT

            # For BC tests
            velocity_bc_left_right = NoSlip((x, t, equations) -> initial_condition_navier_stokes_convergence_test(x,
                                                                                                                  t,
                                                                                                                  equations)[2:3])
            heat_bc_left = Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                              t,
                                                                                                                              equations),
                                                                             equations_parabolic))
            heat_bc_right = Adiabatic((x, t, equations) -> oftype(t, 0))

            boundary_condition_left = BoundaryConditionNavierStokesWall(velocity_bc_left_right,
                                                                        heat_bc_left)
            boundary_condition_right = BoundaryConditionNavierStokesWall(velocity_bc_left_right,
                                                                         heat_bc_right)

            # BC tests
            @test eltype(@inferred boundary_condition_right(flux_inner, w_inner,
                                                            normal,
                                                            x,
                                                            t,
                                                            operator_gradient,
                                                            equations_parabolic)) ==
                  RealT
            @test eltype(@inferred boundary_condition_right(flux_inner, w_inner,
                                                            normal,
                                                            x,
                                                            t,
                                                            operator_divergence,
                                                            equations_parabolic)) ==
                  RealT
            @test eltype(@inferred boundary_condition_left(flux_inner, w_inner,
                                                           normal,
                                                           x,
                                                           t, operator_gradient,
                                                           equations_parabolic)) ==
                  RealT
            @test eltype(@inferred boundary_condition_left(flux_inner, w_inner,
                                                           normal,
                                                           x,
                                                           t, operator_divergence,
                                                           equations_parabolic)) ==
                  RealT
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32,
                                              equations_parabolic_primitive)
        @test adapted isa CompressibleNavierStokesDiffusion2D
        @test typeof(adapted.mu) == Float32
        @test typeof(adapted.Pr) == Float32
        @test typeof(adapted.kappa) == Float32
        @test adapted.equations_hyperbolic isa CompressibleEulerEquations2D{Float32}
    end
end

@testitem "Type stability: Compressible Navier Stokes Diffusion 3D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations3D(RealT(1.4))
        prandtl_number = RealT(0.72)
        mu = RealT(0.01)
        equations_parabolic_primitive = @inferred CompressibleNavierStokesDiffusion3D(equations,
                                                                                      mu = mu,
                                                                                      Prandtl = prandtl_number,
                                                                                      gradient_variables = GradientVariablesPrimitive())
        equations_parabolic_entropy = @inferred CompressibleNavierStokesDiffusion3D(equations,
                                                                                    mu = mu,
                                                                                    Prandtl = prandtl_number,
                                                                                    gradient_variables = GradientVariablesEntropy())

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        t = zero(RealT)
        u = w_inner = u_transformed = flux_inner = normal = SVector(one(RealT),
                                                                    zero(RealT),
                                                                    zero(RealT),
                                                                    zero(RealT),
                                                                    zero(RealT))
        orientations = [1, 2, 3]
        gradient = SVector(RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1))
        gradients = SVector(gradient, gradient, gradient)

        operator_gradient = Trixi.Gradient()
        operator_divergence = Trixi.Divergence()

        # For BC tests
        function initial_condition_navier_stokes_convergence_test(x, t, equations)
            RealT_local = eltype(x)
            c = 2
            A1 = 0.5f0
            A2 = 1
            A3 = 0.5f0

            pi_x = convert(RealT_local, pi) * x[1]
            pi_y = convert(RealT_local, pi) * x[2]
            pi_z = convert(RealT_local, pi) * x[3]
            pi_t = convert(RealT_local, pi) * t

            rho = c + A1 * sin(pi_x) * cos(pi_y) * sin(pi_z) * cos(pi_t)
            v1 = A2 * sin(pi_x) * log(x[2] + 2) * (1 - exp(-A3 * (x[2] - 1))) *
                 sin(pi_z) * cos(pi_t)
            v2 = v1
            v3 = v1
            p = rho^2

            return prim2cons(SVector(rho, v1, v2, v3, p), equations)
        end

        for equations_parabolic in (equations_parabolic_primitive,
                                    equations_parabolic_entropy)
            for orientation in orientations
                @test eltype(@inferred flux(u, gradients, orientation,
                                            equations_parabolic)) == RealT
            end

            @test eltype(@inferred cons2prim(u, equations_parabolic)) == RealT
            @test eltype(@inferred prim2cons(u, equations_parabolic)) == RealT
            @test eltype(@inferred cons2entropy(u, equations_parabolic)) == RealT
            @test eltype(@inferred entropy2cons(u, equations_parabolic)) == RealT
            @test typeof(@inferred Trixi.temperature(u, equations_parabolic)) == RealT
            if equations_parabolic.gradient_variables isa GradientVariablesEntropy
                w = cons2entropy(u, equations_parabolic)
                @test eltype(@inferred Trixi.entropy2velocity_temperature(w,
                                                                          equations_parabolic)) ==
                      RealT
            end
            @test typeof(@inferred Trixi.enstrophy(u, gradients, equations_parabolic)) ==
                  RealT
            @test eltype(@inferred Trixi.vorticity(u, gradients, equations_parabolic)) ==
                  RealT

            @test eltype(@inferred Trixi.convert_transformed_to_velocity_temperature(u_transformed,
                                                                                     equations_parabolic)) ==
                  RealT
            @test eltype(@inferred Trixi.convert_derivative_to_primitive(u, gradient,
                                                                         equations_parabolic)) ==
                  RealT

            # For BC tests
            velocity_bc_left_right = NoSlip((x, t, equations) -> initial_condition_navier_stokes_convergence_test(x,
                                                                                                                  t,
                                                                                                                  equations)[2:4])
            heat_bc_left = Isothermal((x, t, equations) -> Trixi.temperature(initial_condition_navier_stokes_convergence_test(x,
                                                                                                                              t,
                                                                                                                              equations),
                                                                             equations_parabolic))
            heat_bc_right = Adiabatic((x, t, equations) -> oftype(t, 0))

            boundary_condition_left = BoundaryConditionNavierStokesWall(velocity_bc_left_right,
                                                                        heat_bc_left)
            boundary_condition_right = BoundaryConditionNavierStokesWall(velocity_bc_left_right,
                                                                         heat_bc_right)

            # BC tests
            @test eltype(@inferred boundary_condition_right(flux_inner, w_inner,
                                                            normal,
                                                            x,
                                                            t,
                                                            operator_gradient,
                                                            equations_parabolic)) ==
                  RealT
            @test eltype(@inferred boundary_condition_right(flux_inner, w_inner,
                                                            normal,
                                                            x,
                                                            t,
                                                            operator_divergence,
                                                            equations_parabolic)) ==
                  RealT
            @test eltype(@inferred boundary_condition_left(flux_inner, w_inner,
                                                           normal,
                                                           x,
                                                           t, operator_gradient,
                                                           equations_parabolic)) ==
                  RealT
            @test eltype(@inferred boundary_condition_left(flux_inner, w_inner,
                                                           normal,
                                                           x,
                                                           t, operator_divergence,
                                                           equations_parabolic)) ==
                  RealT
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32,
                                              equations_parabolic_primitive)
        @test adapted isa CompressibleNavierStokesDiffusion3D
        @test typeof(adapted.mu) == Float32
        @test typeof(adapted.Pr) == Float32
        @test typeof(adapted.kappa) == Float32
        @test adapted.equations_hyperbolic isa CompressibleEulerEquations3D{Float32}
    end
end

@testitem "Type stability: Testing Trixi.entropy2velocity_temperature for CompressibleNavierStokesDiffusion" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        prandtl_number = RealT(0.72)
        mu = RealT(0.01)

        equations_1d = CompressibleEulerEquations1D(RealT(1.4))
        equations_parabolic_1d = CompressibleNavierStokesDiffusion1D(equations_1d,
                                                                     mu = mu,
                                                                     Prandtl = prandtl_number,
                                                                     gradient_variables = GradientVariablesEntropy())
        u_1d = prim2cons(SVector(RealT(2.0), RealT(0.1), RealT(4.0)), equations_1d)
        w_1d = cons2entropy(u_1d, equations_parabolic_1d)
        @test Trixi.entropy2velocity_temperature(w_1d, equations_parabolic_1d) ≈
              cons2prim(u_1d, equations_parabolic_1d)[2:end]
        @test length(Trixi.entropy2velocity_temperature(w_1d, equations_parabolic_1d)) ==
              2

        equations_2d = CompressibleEulerEquations2D(RealT(1.4))
        equations_parabolic_2d = CompressibleNavierStokesDiffusion2D(equations_2d,
                                                                     mu = mu,
                                                                     Prandtl = prandtl_number,
                                                                     gradient_variables = GradientVariablesEntropy())
        u_2d = prim2cons(SVector(RealT(2.0), RealT(0.1), RealT(0.2), RealT(4.0)),
                         equations_2d)
        w_2d = cons2entropy(u_2d, equations_parabolic_2d)
        @test Trixi.entropy2velocity_temperature(w_2d, equations_parabolic_2d) ≈
              cons2prim(u_2d, equations_parabolic_2d)[2:end]
        @test length(Trixi.entropy2velocity_temperature(w_2d, equations_parabolic_2d)) ==
              3

        equations_3d = CompressibleEulerEquations3D(RealT(1.4))
        equations_parabolic_3d = CompressibleNavierStokesDiffusion3D(equations_3d,
                                                                     mu = mu,
                                                                     Prandtl = prandtl_number,
                                                                     gradient_variables = GradientVariablesEntropy())
        u_3d = prim2cons(SVector(RealT(2.0), RealT(0.1), RealT(0.2), RealT(0.3),
                                 RealT(4.0)), equations_3d)
        w_3d = cons2entropy(u_3d, equations_parabolic_3d)
        @test Trixi.entropy2velocity_temperature(w_3d, equations_parabolic_3d) ≈
              cons2prim(u_3d, equations_parabolic_3d)[2:end]
        @test length(Trixi.entropy2velocity_temperature(w_3d, equations_parabolic_3d)) ==
              4
    end
end

@testitem "Type stability: Hyperbolic Diffusion 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        nu = one(RealT)
        Lr = RealT(inv(2pi))
        equations = @inferred HyperbolicDiffusionEquations1D(nu = nu, Lr = Lr)

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = du = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT))
        orientation = 1
        directions = [1, 2]

        surface_flux_function = flux_lax_friedrichs

        @test typeof(@inferred Trixi.residual_steady_state(du, equations)) == RealT
        @test eltype(@inferred initial_condition_poisson_nonperiodic(x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_poisson_nonperiodic(u, x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_harmonic(u, x, t, equations)) == RealT
        @test eltype(@inferred Trixi.initial_condition_eoc_test_coupled_euler_gravity(x,
                                                                                      t,
                                                                                      equations)) ==
              RealT

        for direction in directions
            @test eltype(@inferred boundary_condition_poisson_nonperiodic(u_inner,
                                                                          orientation,
                                                                          direction,
                                                                          x, t,
                                                                          surface_flux_function,
                                                                          equations)) ==
                  RealT
        end

        @test eltype(@inferred flux(u, orientation, equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                   equations)) == RealT
        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred energy_total(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa HyperbolicDiffusionEquations1D{Float32}
        @test typeof(adapted.nu) == Float32
        @test typeof(adapted.Lr) == Float32
    end
end

@testitem "Type stability: Hyperbolic Diffusion 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        nu = one(RealT)
        Lr = RealT(inv(2pi))
        equations = @inferred HyperbolicDiffusionEquations2D(nu = nu, Lr = Lr)

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = du = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT))
        orientations = [1, 2]
        directions = [1, 2, 3, 4]
        normal_direction = SVector(one(RealT), zero(RealT))

        surface_flux_function = flux_lax_friedrichs

        @test typeof(@inferred Trixi.residual_steady_state(du, equations)) == RealT
        @test eltype(@inferred initial_condition_poisson_nonperiodic(x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_poisson_nonperiodic(u, x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_harmonic(u, x, t, equations)) == RealT
        @test eltype(@inferred Trixi.initial_condition_eoc_test_coupled_euler_gravity(x,
                                                                                      t,
                                                                                      equations)) ==
              RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred boundary_condition_poisson_nonperiodic(u_inner,
                                                                              orientation,
                                                                              direction,
                                                                              x, t,
                                                                              surface_flux_function,
                                                                              equations)) ==
                      RealT
            end
        end

        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_godunov(u_ll, u_rr, normal_direction, equations)) ==
              RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation,
                                                equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
        end

        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred energy_total(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa HyperbolicDiffusionEquations2D{Float32}
        @test typeof(adapted.nu) == Float32
        @test typeof(adapted.Lr) == Float32
    end
end

@testitem "Type stability: Hyperbolic Diffusion 3D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        nu = one(RealT)
        Lr = RealT(inv(2pi))
        equations = @inferred HyperbolicDiffusionEquations3D(nu = nu, Lr = Lr)

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        t = zero(RealT)
        u = du = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT),
                                                 one(RealT))
        orientations = [1, 2, 3]
        directions = [1, 2, 3, 4, 5, 6]

        surface_flux_function = flux_lax_friedrichs

        @test typeof(@inferred Trixi.residual_steady_state(du, equations)) == RealT
        @test eltype(@inferred initial_condition_poisson_nonperiodic(x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_poisson_nonperiodic(u, x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_harmonic(u, x, t, equations)) == RealT
        @test eltype(@inferred Trixi.initial_condition_eoc_test_coupled_euler_gravity(x,
                                                                                      t,
                                                                                      equations)) ==
              RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred boundary_condition_poisson_nonperiodic(u_inner,
                                                                              orientation,
                                                                              direction,
                                                                              x, t,
                                                                              surface_flux_function,
                                                                              equations)) ==
                      RealT
            end
        end

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation,
                                                equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
        end

        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred energy_total(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa HyperbolicDiffusionEquations3D{Float32}
        @test typeof(adapted.nu) == Float32
        @test typeof(adapted.Lr) == Float32
    end
end

@testitem "Type stability: Ideal Glm Mhd 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred IdealGlmMhdEquations1D(RealT(1.4))

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = cons = SVector(one(RealT), zero(RealT), zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT))
        orientation = 1
        directions = [1, 2]

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        @test eltype(@inferred velocity(u, equations)) == RealT
        @test eltype(@inferred flux(u, orientation, equations)) == RealT
        @test eltype(@inferred flux_hllc(u_ll, u_rr, orientation, equations)) == RealT
        @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                       orientation,
                                                       equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                      equations)) ==
              RealT
        @test typeof(@inferred Trixi.max_abs_speeds(u, equations)) ==
              RealT
        @test eltype(@inferred cons2prim(u, equations)) ==
              RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT

        for direction in directions
            @test typeof(Trixi.calc_fast_wavespeed(cons, direction, equations)) == RealT
            @test eltype(Trixi.calc_fast_wavespeed_roe(u_ll, u_rr, direction,
                                                       equations)) == RealT
        end

        @test typeof(@inferred entropy_thermodynamic(cons, equations)) == RealT
        @test typeof(@inferred entropy_math(cons, equations)) == RealT
        @test typeof(@inferred entropy(cons, equations)) == RealT
        @test typeof(@inferred energy_total(cons, equations)) == RealT
        @test typeof(@inferred energy_kinetic(cons, equations)) == RealT
        @test typeof(@inferred energy_magnetic(cons, equations)) == RealT
        @test typeof(@inferred energy_internal(cons, equations)) == RealT
        @test typeof(@inferred cross_helicity(cons, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa IdealGlmMhdEquations1D{Float32}
        @test typeof(adapted.gamma) == Float32
    end
end

@testitem "Type stability: Ideal Glm Mhd 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred IdealGlmMhdEquations2D(RealT(1.4))

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = cons = SVector(one(RealT), zero(RealT), zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT))
        orientations = [1, 2]
        directions = [1, 2, 3, 4]
        normal_direction = SVector(one(RealT), zero(RealT))
        nonconservative_type_local = Trixi.NonConservativeLocal()
        nonconservative_type_symmetric = Trixi.NonConservativeSymmetric()
        nonconservative_terms = [1, 2]

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        @test eltype(@inferred velocity(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr,
                                                           normal_direction,
                                                           equations)) == RealT
        @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, normal_direction,
                                                      equations)) == RealT

        for orientation in orientations
            @test eltype(@inferred velocity(u, orientation, equations)) == RealT
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr, orientation,
                                                               equations)) == RealT
            @test eltype(@inferred flux_nonconservative_powell_local_symmetric(u_ll,
                                                                               u_rr,
                                                                               orientation,
                                                                               equations)) ==
                  RealT
            @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                    equations)) == RealT
            @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, orientation,
                                                           equations)) == RealT
            for nonconservative_term in nonconservative_terms
                @test eltype(@inferred flux_nonconservative_powell_local_symmetric(u_ll,
                                                                                   orientation,
                                                                                   equations,
                                                                                   nonconservative_type_local,
                                                                                   nonconservative_term)) ==
                      RealT
                @test eltype(@inferred flux_nonconservative_powell_local_symmetric(u_ll,
                                                                                   u_rr,
                                                                                   orientation,
                                                                                   equations,
                                                                                   nonconservative_type_symmetric,
                                                                                   nonconservative_term)) ==
                      RealT
            end

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                          equations)) == RealT
        end

        @test eltype(@inferred velocity(u, equations)) == RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred entropy2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT
        @test eltype(@inferred Trixi.gradient_conservative(pressure, u, equations)) ==
              RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT

        @test typeof(@inferred Trixi.calc_fast_wavespeed(cons, normal_direction,
                                                         equations)) == RealT
        @test eltype(@inferred Trixi.calc_fast_wavespeed_roe(u_ll, u_rr,
                                                             normal_direction,
                                                             equations)) == RealT
        for orientation in orientations
            @test typeof(@inferred Trixi.calc_fast_wavespeed(cons, orientation,
                                                             equations)) ==
                  RealT
            @test eltype(@inferred Trixi.calc_fast_wavespeed_roe(u_ll, u_rr,
                                                                 orientation,
                                                                 equations)) == RealT
        end

        @test typeof(@inferred entropy_thermodynamic(cons, equations)) == RealT
        @test typeof(@inferred entropy_math(cons, equations)) == RealT
        @test typeof(@inferred entropy(cons, equations)) == RealT
        @test typeof(@inferred energy_total(cons, equations)) == RealT
        @test typeof(@inferred energy_kinetic(cons, equations)) == RealT
        @test typeof(@inferred energy_magnetic(cons, equations)) == RealT
        @test typeof(@inferred energy_internal(cons, equations)) == RealT
        @test typeof(@inferred cross_helicity(cons, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa IdealGlmMhdEquations2D{Float32}
        @test typeof(adapted.gamma) == Float32
        @test typeof(adapted.c_h) == Float32
    end
end

@testitem "Type stability: Ideal Glm Mhd 3D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred IdealGlmMhdEquations3D(RealT(1.4))

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = cons = SVector(one(RealT), zero(RealT), zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT),
                                         zero(RealT))
        orientations = [1, 2, 3]
        directions = [1, 2, 3, 4, 5, 6]
        normal_direction = SVector(one(RealT), zero(RealT), zero(RealT))

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        @test eltype(@inferred velocity(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr,
                                                           normal_direction,
                                                           equations)) == RealT
        @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, normal_direction,
                                                      equations)) == RealT

        for orientation in orientations
            @test eltype(@inferred velocity(u, orientation, equations)) == RealT
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr, orientation,
                                                               equations)) == RealT
            @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                    equations)) == RealT
            @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, orientation,
                                                           equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                          equations)) == RealT
        end

        @test eltype(@inferred velocity(u, equations)) == RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT

        @test typeof(@inferred Trixi.calc_fast_wavespeed(cons, normal_direction,
                                                         equations)) == RealT
        @test eltype(@inferred Trixi.calc_fast_wavespeed_roe(u_ll, u_rr,
                                                             normal_direction,
                                                             equations)) == RealT
        for orientation in orientations
            @test typeof(@inferred Trixi.calc_fast_wavespeed(cons, orientation,
                                                             equations)) ==
                  RealT
            @test eltype(@inferred Trixi.calc_fast_wavespeed_roe(u_ll, u_rr,
                                                                 orientation,
                                                                 equations)) == RealT
        end

        @test typeof(@inferred entropy_thermodynamic(cons, equations)) == RealT
        @test typeof(@inferred entropy_math(cons, equations)) == RealT
        @test typeof(@inferred entropy(cons, equations)) == RealT
        @test typeof(@inferred energy_total(cons, equations)) == RealT
        @test typeof(@inferred energy_kinetic(cons, equations)) == RealT
        @test typeof(@inferred energy_magnetic(cons, equations)) == RealT
        @test typeof(@inferred energy_internal(cons, equations)) == RealT
        @test typeof(@inferred cross_helicity(cons, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa IdealGlmMhdEquations3D{Float32}
        @test typeof(adapted.gamma) == Float32
        @test typeof(adapted.c_h) == Float32
    end
end

@testitem "Type stability: Ideal Glm Mhd Multicomponent 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        gammas = (RealT(2), RealT(2))
        gas_constants = (RealT(2), RealT(2))
        equations = @inferred IdealGlmMhdMulticomponentEquations1D(gammas = gammas,
                                                                   gas_constants = gas_constants)

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = cons = SVector(one(RealT), one(RealT), one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT))
        orientation = 1
        directions = [1, 2]

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        @test eltype(@inferred flux(u, orientation, equations)) == RealT
        @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, orientation,
                                                       equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) ==
              RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT
        @test typeof(@inferred Trixi.totalgamma(u, equations)) == RealT

        for direction in directions
            @test typeof(Trixi.calc_fast_wavespeed(cons, direction, equations)) == RealT
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa IdealGlmMhdMulticomponentEquations1D
        @test eltype(adapted.gammas) == Float32
        @test eltype(adapted.cv) == Float32
    end
end

@testitem "Type stability: Ideal Glm Mhd Multicomponent 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        gammas = (RealT(2), RealT(2))
        gas_constants = (RealT(2), RealT(2))
        equations = @inferred IdealGlmMhdMulticomponentEquations2D(gammas = gammas,
                                                                   gas_constants = gas_constants)

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = cons = SVector(one(RealT), one(RealT), one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT))
        orientations = [1, 2]
        directions = [1, 2, 3, 4]

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr, orientation,
                                                               equations)) ==
                  RealT
            @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                    equations)) ==
                  RealT
            @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, orientation,
                                                           equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
        end

        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT
        @test typeof(@inferred Trixi.totalgamma(u, equations)) == RealT

        for direction in directions
            @test typeof(Trixi.calc_fast_wavespeed(cons, direction, equations)) == RealT
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa IdealGlmMhdMulticomponentEquations2D
        @test eltype(adapted.gammas) == Float32
        @test eltype(adapted.cv) == Float32
        @test typeof(adapted.c_h) == Float32
    end
end

@testitem "Type stability: Ideal Glm Mhd MultiIon 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        gammas = (RealT(2), RealT(2))
        charge_to_mass = (RealT(2), RealT(2))
        equations = @inferred IdealGlmMhdMultiIonEquations2D(gammas = gammas,
                                                             charge_to_mass = charge_to_mass)

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = cons = SVector(one(RealT), one(RealT), one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT))
        dissipation_es = DissipationLaxFriedrichsEntropyVariables()
        orientations = [1, 2]
        normal_direction = SVector(one(RealT), zero(RealT))

        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        @test eltype(@inferred source_terms_lorentz(u, x, t, equations)) ==
              RealT

        @test eltype(@inferred source_terms_collision_ion_ion(u, x, t, equations)) ==
              RealT

        @test eltype(@inferred source_terms_collision_ion_electron(u, x, t, equations)) ==
              RealT

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                                                          orientation,
                                                                          equations)) ==
                  RealT
            @test eltype(@inferred flux_nonconservative_central(u_ll, u_rr, orientation,
                                                                equations)) ==
                  RealT
            @test eltype(@inferred flux_ruedaramirez_etal(u_ll, u_rr, orientation,
                                                          equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred dissipation_es(u_ll, u_rr, orientation, equations)) ==
                  RealT
        end

        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                                                      normal_direction,
                                                                      equations)) ==
              RealT
        @test eltype(@inferred flux_nonconservative_central(u_ll, u_rr,
                                                            normal_direction,
                                                            equations)) == RealT
        @test eltype(@inferred flux_ruedaramirez_etal(u_ll, u_rr, normal_direction,
                                                      equations)) == RealT
        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test typeof(Trixi.calc_fast_wavespeed(cons, normal_direction, equations)) ==
              RealT

        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test eltype(@inferred magnetic_field(u, equations)) == RealT
        @test typeof(@inferred divergence_cleaning_field(u, equations)) == RealT
        @test typeof(@inferred Trixi.electron_pressure_zero(u, equations)) == RealT

        @test typeof(@inferred Trixi.charge_averaged_velocities(u, equations)) ==
              Tuple{RealT, RealT, RealT, SVector{2, RealT}, SVector{2, RealT},
                    SVector{2, RealT}}

        for k in 1:2
            @test eltype(@inferred Trixi.get_component(k, u, equations)) == RealT
        end

        for direction in orientations
            @test typeof(Trixi.calc_fast_wavespeed(cons, direction, equations)) == RealT
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa IdealGlmMhdMultiIonEquations2D
        @test eltype(adapted.gammas) == Float32
        @test eltype(adapted.ion_ion_collision_constants) == Float32
        @test typeof(adapted.c_h) == Float32
    end
end

@testitem "Type stability: Ideal Glm Mhd MultiIon 3D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        gammas = (RealT(2), RealT(2))
        charge_to_mass = (RealT(2), RealT(2))
        equations = @inferred IdealGlmMhdMultiIonEquations3D(gammas = gammas,
                                                             charge_to_mass = charge_to_mass)

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        normal_direction = SVector(one(RealT), zero(RealT), zero(RealT))

        t = zero(RealT)
        u = u_ll = u_rr = cons = SVector(one(RealT), one(RealT), one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT),
                                         one(RealT))
        dissipation_es = DissipationLaxFriedrichsEntropyVariables()
        orientations = [1, 2, 3]

        @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
              RealT

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                                                          orientation,
                                                                          equations)) ==
                  RealT
            @test eltype(@inferred flux_nonconservative_central(u_ll, u_rr, orientation,
                                                                equations)) ==
                  RealT
            @test eltype(@inferred flux_ruedaramirez_etal(u_ll, u_rr, orientation,
                                                          equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
        end
        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_nonconservative_ruedaramirez_etal(u_ll, u_rr,
                                                                      normal_direction,
                                                                      equations)) ==
              RealT
        @test eltype(@inferred flux_ruedaramirez_etal(u_ll, u_rr, normal_direction,
                                                      equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) ==
              RealT

        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT

        for direction in orientations
            @test typeof(Trixi.calc_fast_wavespeed(cons, direction, equations)) == RealT
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa IdealGlmMhdMultiIonEquations3D
        @test eltype(adapted.gammas) == Float32
        @test eltype(adapted.charge_to_mass) == Float32
        @test typeof(adapted.c_h) == Float32
    end
end

@testitem "Type stability: Inviscid Burgers 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred InviscidBurgersEquation1D()

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = SVector(one(RealT))
        orientation = 1

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT

        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT

        @test eltype(@inferred flux(u, orientation, equations)) == RealT
        @test eltype(@inferred flux_ec(u_ll, u_rr, orientation, equations)) == RealT
        @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.flux_engquist_osher(u_ll, u_rr, orientation,
                                                         equations)) ==
              RealT

        @test eltype(eltype(@inferred splitting_lax_friedrichs(u, orientation,
                                                               equations))) ==
              RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) ==
              RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test eltype(@inferred entropy2cons(u, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred energy_total(u, equations)) == RealT
    end
end

@testitem "Type stability: Laplace Diffusion 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LinearScalarAdvectionEquation1D(RealT(1))
        equations_parabolic = @inferred LaplaceDiffusion1D(RealT(0.1), equations)

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_inner = flux_inner = normal = gradients = SVector(one(RealT))
        orientation = 1

        operator_gradient = Trixi.Gradient()
        operator_divergence = Trixi.Divergence()

        @test eltype(@inferred flux(u, (gradients,), orientation, equations_parabolic)) ==
              RealT

        # For BC tests
        function initial_condition_convergence_test(x, t,
                                                    equation::LaplaceDiffusion1D)
            RealT_local = eltype(x)
            x_trans = x[1] - equation.diffusivity * t

            c = 1
            A = 0.5f0
            L = 2
            f = 1.0f0 / L
            omega = 2 * convert(RealT_local, pi) * f
            scalar = c + A * sin(omega * sum(x_trans))
            return SVector(scalar)
        end

        boundary_condition_dirichlet = BoundaryConditionDirichlet(initial_condition_convergence_test)
        boundary_condition_neumann = BoundaryConditionNeumann((x, t, equations) -> oftype(t,
                                                                                          0))

        # BC tests
        @test eltype(@inferred boundary_condition_dirichlet(flux_inner, u_inner, normal,
                                                            x, t,
                                                            operator_gradient,
                                                            equations_parabolic)) ==
              RealT
        @test eltype(@inferred boundary_condition_dirichlet(flux_inner, u_inner, normal,
                                                            x, t,
                                                            operator_divergence,
                                                            equations_parabolic)) ==
              RealT
        @test eltype(@inferred boundary_condition_neumann(flux_inner, u_inner, normal,
                                                          x, t,
                                                          operator_gradient,
                                                          equations_parabolic)) == RealT
        @test eltype(@inferred boundary_condition_neumann(flux_inner, u_inner, normal,
                                                          x, t,
                                                          operator_divergence,
                                                          equations_parabolic)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations_parabolic)
        @test adapted isa LaplaceDiffusion1D
        @test typeof(adapted.diffusivity) == Float32
        @test adapted.equations_hyperbolic isa LinearScalarAdvectionEquation1D{Float32}
    end
end

@testitem "Type stability: Linear Diffusion Equation" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        u = SVector(one(RealT))

        equations_1d = LinearDiffusionEquation1D(RealT(0.1))
        @test eltype(@inferred cons2prim(u, equations_1d)) == RealT
        @test eltype(@inferred cons2entropy(u, equations_1d)) == RealT

        equations_2d = LinearDiffusionEquation2D(RealT(0.1))
        @test eltype(@inferred cons2prim(u, equations_2d)) == RealT
        @test eltype(@inferred cons2entropy(u, equations_2d)) == RealT

        adapted_1d = @inferred Trixi.trixi_adapt(Array, Float32, equations_1d)
        @test adapted_1d isa LinearDiffusionEquation1D{Float32}
        @test typeof(adapted_1d.diffusivity) == Float32
        adapted_2d = @inferred Trixi.trixi_adapt(Array, Float32, equations_2d)
        @test adapted_2d isa LinearDiffusionEquation2D{Float32}
        @test typeof(adapted_2d.diffusivity) == Float32
    end
end

@testitem "Type stability: Laplace Diffusion Entropy Variables 1D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations1D(RealT(1.4))
        equations_parabolic = @inferred LaplaceDiffusionEntropyVariables1D(RealT(0.01),
                                                                           equations)

        prim = SVector(RealT(1), RealT(0.2), RealT(2))
        w = cons2entropy(prim2cons(prim, equations), equations)
        gradient = SVector(RealT(0.1), RealT(0.1), RealT(0.1))
        gradients = (gradient,)

        @test eltype(@inferred flux(w, gradients, 1, equations_parabolic)) == RealT
    end
end

@testitem "Type stability: Laplace Diffusion Entropy Variables 2D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations2D(RealT(1.4))
        equations_parabolic = @inferred LaplaceDiffusionEntropyVariables2D(RealT(0.01),
                                                                           equations)

        prim = SVector(RealT(1), RealT(0.2), RealT(0.2), RealT(2))
        w = cons2entropy(prim2cons(prim, equations), equations)
        gradient = SVector(RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1))
        gradients = SVector(gradient, gradient)

        for orientation in (1, 2)
            @test eltype(@inferred flux(w, gradients, orientation, equations_parabolic)) ==
                  RealT
        end
    end
end

@testitem "Type stability: Laplace Diffusion Entropy Variables 3D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations3D(RealT(1.4))
        equations_parabolic = @inferred LaplaceDiffusionEntropyVariables3D(RealT(0.01),
                                                                           equations)

        prim = SVector(RealT(1), RealT(0.2), RealT(0.2), RealT(0.1), RealT(2))
        w = cons2entropy(prim2cons(prim, equations), equations)
        gradient = SVector(RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1))
        gradients = SVector(gradient, gradient, gradient)

        for orientation in (1, 2, 3)
            @test eltype(@inferred flux(w, gradients, orientation, equations_parabolic)) ==
                  RealT
        end
    end
end

@testitem "Type stability: Laplace Diffusion 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = LinearScalarAdvectionEquation2D(RealT(1), RealT(1))
        equations_parabolic = LaplaceDiffusion2D(RealT(0.1), equations)

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_inner = u_outer = inv_h = gradients = SVector(one(RealT), one(RealT))
        orientations = [1, 2]

        for orientation in orientations
            @test eltype(@inferred flux(u, gradients, orientation, equations_parabolic)) ==
                  RealT
        end

        parabolic_solver = ParabolicFormulationLocalDG(RealT(0.1))
        @test eltype(@inferred Trixi.penalty(u_outer, u_inner, inv_h,
                                             equations_parabolic, parabolic_solver)) ==
              RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations_parabolic)
        @test adapted isa LaplaceDiffusion2D
        @test typeof(adapted.diffusivity) == Float32
        @test adapted.equations_hyperbolic isa LinearScalarAdvectionEquation2D{Float32}
    end
end

@testitem "Type stability: Laplace Diffusion 3D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = LinearScalarAdvectionEquation3D(RealT(1), RealT(1), RealT(1))
        equations_parabolic = LaplaceDiffusion3D(RealT(0.1), equations)

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_inner = u_outer = inv_h = gradients = SVector(one(RealT), one(RealT),
                                                            one(RealT))
        orientations = [1, 2, 3]

        for orientation in orientations
            @test eltype(@inferred flux(u, gradients, orientation, equations_parabolic)) ==
                  RealT
        end

        parabolic_solver = ParabolicFormulationLocalDG(RealT(0.1))
        @test eltype(@inferred Trixi.penalty(u_outer, u_inner, inv_h,
                                             equations_parabolic, parabolic_solver)) ==
              RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations_parabolic)
        @test adapted isa LaplaceDiffusion3D
        @test typeof(adapted.diffusivity) == Float32
        @test adapted.equations_hyperbolic isa LinearScalarAdvectionEquation3D{Float32}
    end
end

@testitem "Type stability: Laplace Diffusion Entropy Variables 1D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations1D(RealT(1.4))
        equations_parabolic = @inferred LaplaceDiffusionEntropyVariables1D(RealT(0.1),
                                                                           equations)

        u = gradients = SVector(one(RealT), zero(RealT), zero(RealT))
        orientation = 1

        @test eltype(@inferred flux(u, (gradients,), orientation, equations_parabolic)) ==
              RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations_parabolic)
        @test adapted isa Trixi.LaplaceDiffusionEntropyVariables{1}
        @test typeof(adapted.diffusivity) == Float32
        @test adapted.equations_hyperbolic isa CompressibleEulerEquations1D{Float32}
    end
end

@testitem "Type stability: Laplace Diffusion Entropy Variables 2D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations2D(RealT(1.4))
        equations_parabolic = @inferred LaplaceDiffusionEntropyVariables2D(RealT(0.1),
                                                                           equations)

        u = gradient = SVector(one(RealT), zero(RealT), zero(RealT), zero(RealT))
        orientations = [1, 2]

        for orientation in orientations
            @test eltype(@inferred flux(u, (gradient, gradient), orientation,
                                        equations_parabolic)) ==
                  RealT
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations_parabolic)
        @test adapted isa Trixi.LaplaceDiffusionEntropyVariables{2}
        @test typeof(adapted.diffusivity) == Float32
        @test adapted.equations_hyperbolic isa CompressibleEulerEquations2D{Float32}
    end
end

@testitem "Type stability: Laplace Diffusion Entropy Variables 3D" setup=[
    Setup,
    TypeStability
] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred CompressibleEulerEquations3D(RealT(1.4))
        equations_parabolic = @inferred LaplaceDiffusionEntropyVariables3D(RealT(0.1),
                                                                           equations)

        u = gradient = SVector(one(RealT), zero(RealT), zero(RealT), zero(RealT),
                               zero(RealT))
        orientations = [1, 2, 3]

        for orientation in orientations
            @test eltype(@inferred flux(u, (gradient, gradient, gradient), orientation,
                                        equations_parabolic)) ==
                  RealT
        end

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations_parabolic)
        @test adapted isa Trixi.LaplaceDiffusionEntropyVariables{3}
        @test typeof(adapted.diffusivity) == Float32
        @test adapted.equations_hyperbolic isa CompressibleEulerEquations3D{Float32}
    end
end

@testitem "Type stability: Lattice Boltzmann 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LatticeBoltzmannEquations2D(Ma = RealT(0.1), Re = 1000)

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT),
                                            one(RealT), one(RealT), one(RealT),
                                            one(RealT), one(RealT), one(RealT))
        orientations = [1, 2]
        directions = [1, 2, 3, 4]
        p = rho = dt = one(RealT)

        surface_flux_function = flux_godunov

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred boundary_condition_noslip_wall(u_inner,
                                                                      orientation,
                                                                      direction, x, t,
                                                                      surface_flux_function,
                                                                      equations)) ==
                      RealT
            end

            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation, equations)) ==
                  RealT

            @test typeof(@inferred velocity(u, orientation, equations)) == RealT
        end

        @test typeof(@inferred density(p, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test eltype(@inferred velocity(u, equations)) == RealT
        @test typeof(@inferred pressure(rho, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT

        @test eltype(@inferred Trixi.collision_bgk(u, dt, equations)) == RealT

        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LatticeBoltzmannEquations2D{Float32}
        @test typeof(adapted.c) == Float32
        @test eltype(adapted.weights) == Float32
        @test eltype(adapted.v_alpha1) == Float32
    end
end

@testitem "Type stability: Lattice Boltzmann 3D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LatticeBoltzmannEquations3D(Ma = RealT(0.1), Re = 1000)

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = SVector(one(RealT), one(RealT), one(RealT),
                                  one(RealT), one(RealT), one(RealT),
                                  one(RealT), one(RealT), one(RealT),
                                  one(RealT), one(RealT), one(RealT),
                                  one(RealT), one(RealT), one(RealT),
                                  one(RealT), one(RealT), one(RealT),
                                  one(RealT), one(RealT), one(RealT),
                                  one(RealT), one(RealT), one(RealT),
                                  one(RealT), one(RealT), one(RealT))
        orientations = [1, 2, 3]
        p = rho = dt = one(RealT)

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation, equations)) ==
                  RealT

            @test typeof(@inferred velocity(u, orientation, equations)) == RealT
        end

        @test typeof(@inferred density(p, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test eltype(@inferred velocity(u, equations)) == RealT
        @test typeof(@inferred pressure(rho, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT

        @test eltype(@inferred Trixi.collision_bgk(u, dt, equations)) == RealT

        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred energy_kinetic(u, equations)) == RealT
        @test typeof(@inferred Trixi.energy_kinetic_nondimensional(u, equations)) ==
              RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LatticeBoltzmannEquations3D{Float32}
        @test typeof(adapted.c) == Float32
        @test eltype(adapted.weights) == Float32
        @test eltype(adapted.v_alpha1) == Float32
    end
end

@testitem "Type stability: Linear Scalar Advection 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LinearScalarAdvectionEquation1D(RealT(1))

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = SVector(one(RealT))
        orientation = 1
        directions = [1, 2]

        surface_flux_function = flux_lax_friedrichs

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_gauss(x, t, equations)) == RealT
        @test eltype(@inferred Trixi.initial_condition_sin(x, t, equations)) == RealT
        @test eltype(@inferred Trixi.initial_condition_linear_x(x, t, equations)) ==
              RealT

        for direction in directions
            @test eltype(@inferred Trixi.boundary_condition_linear_x(u_inner,
                                                                     orientation,
                                                                     direction, x,
                                                                     t,
                                                                     surface_flux_function,
                                                                     equations)) ==
                  RealT
        end

        @test eltype(@inferred flux(u, orientation, equations)) == RealT
        @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.flux_engquist_osher(u_ll, u_rr, orientation,
                                                         equations)) == RealT

        @test eltype(eltype(@inferred splitting_lax_friedrichs(u, orientation,
                                                               equations))) ==
              RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                   equations)) == RealT
        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred energy_total(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LinearScalarAdvectionEquation1D{Float32}
        @test eltype(adapted.advection_velocity) == Float32
    end
end

@testitem "Type stability: Linear Scalar Advection 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LinearScalarAdvectionEquation2D(RealT(1), RealT(1))

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = SVector(one(RealT))
        orientations = [1, 2]
        directions = [1, 2, 3, 4]
        normal_direction = SVector(one(RealT), zero(RealT))

        surface_flux_function = flux_lax_friedrichs

        @test eltype(@inferred Trixi.x_trans_periodic_2d(x)) == RealT

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_gauss(x, t, equations)) == RealT
        @test eltype(@inferred Trixi.initial_condition_sin_sin(x, t, equations)) ==
              RealT
        @test eltype(@inferred Trixi.initial_condition_linear_x_y(x, t, equations)) ==
              RealT
        @test eltype(@inferred Trixi.initial_condition_linear_x(x, t, equations)) ==
              RealT
        @test eltype(@inferred Trixi.initial_condition_linear_y(x, t, equations)) ==
              RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred Trixi.boundary_condition_linear_x_y(u_inner,
                                                                           orientation,
                                                                           direction,
                                                                           x,
                                                                           t,
                                                                           surface_flux_function,
                                                                           equations)) ==
                      RealT
                @test eltype(@inferred Trixi.boundary_condition_linear_x(u_inner,
                                                                         orientation,
                                                                         direction,
                                                                         x,
                                                                         t,
                                                                         surface_flux_function,
                                                                         equations)) ==
                      RealT
                @test eltype(@inferred Trixi.boundary_condition_linear_y(u_inner,
                                                                         orientation,
                                                                         direction,
                                                                         x,
                                                                         t,
                                                                         surface_flux_function,
                                                                         equations)) ==
                      RealT
            end
        end

        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_godunov(u_ll, u_rr, normal_direction, equations)) ==
              RealT

        @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) ==
              RealT

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation, equations)) ==
                  RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
        end

        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred energy_total(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LinearScalarAdvectionEquation2D{Float32}
        @test eltype(adapted.advection_velocity) == Float32
    end
end

@testitem "Type stability: Linear Scalar Advection 3D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LinearScalarAdvectionEquation3D(RealT(1), RealT(1),
                                                              RealT(1))

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = SVector(one(RealT))
        orientations = [1, 2, 3]
        directions = [1, 2, 3, 4, 5, 6]
        normal_direction = SVector(one(RealT), zero(RealT), zero(RealT))

        surface_flux_function = flux_lax_friedrichs

        @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred initial_condition_gauss(x, t, equations)) == RealT
        @test eltype(@inferred Trixi.initial_condition_sin(x, t, equations)) == RealT
        @test eltype(@inferred Trixi.initial_condition_linear_z(x, t, equations)) ==
              RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred Trixi.boundary_condition_linear_z(u_inner,
                                                                         orientation,
                                                                         direction,
                                                                         x,
                                                                         t,
                                                                         surface_flux_function,
                                                                         equations)) ==
                      RealT
            end
        end

        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_godunov(u_ll, u_rr, normal_direction, equations)) ==
              RealT

        @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT

        for orientation in orientations
            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation, equations)) ==
                  RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
        end

        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred energy_total(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LinearScalarAdvectionEquation3D{Float32}
        @test eltype(adapted.advection_velocity) == Float32
    end
end

@testitem "Type stability: Maxwell 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        c = RealT(299_792_458)
        equations = @inferred MaxwellEquations1D(c)

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = SVector(one(RealT), one(RealT))
        orientation = 1

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT

        @test eltype(@inferred flux(u, orientation, equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.max_abs_speeds(equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa MaxwellEquations1D{Float32}
        @test typeof(adapted.speed_of_light) == Float32
    end
end

@testitem "Type stability: Linearized Euler 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LinearizedEulerEquations1D(v_mean_global = RealT(0),
                                                         c_mean_global = RealT(1),
                                                         rho_mean_global = RealT(1))

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT))
        orientation = 1
        directions = [1, 2]

        surface_flux_function = flux_hll

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT

        for direction in directions
            @test eltype(@inferred boundary_condition_wall(u_inner, orientation,
                                                           direction, x, t,
                                                           surface_flux_function,
                                                           equations)) == RealT
        end

        @test eltype(@inferred flux(u, orientation, equations)) == RealT

        @test typeof(@inferred Trixi.max_abs_speeds(equations)) ==
              RealT
        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation, equations)) ==
              RealT

        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LinearizedEulerEquations1D{Float32}
        @test typeof(adapted.v_mean_global) == Float32
        @test typeof(adapted.c_mean_global) == Float32
    end
end

@testitem "Type stability: Linearized Euler 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LinearizedEulerEquations2D(v_mean_global = (RealT(0),
                                                                          RealT(0)),
                                                         c_mean_global = RealT(1),
                                                         rho_mean_global = RealT(1))

        x = SVector(zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT),
                                            one(RealT))
        orientations = [1, 2]
        directions = [1, 2, 3, 4]
        normal_direction = SVector(one(RealT), zero(RealT))

        surface_flux_function = flux_hll

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred boundary_condition_wall(u_inner, orientation,
                                                               direction, x, t,
                                                               surface_flux_function,
                                                               equations)) == RealT
            end

            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_godunov(u_ll, u_rr, orientation, equations)) ==
                  RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                       equations)) ==
                  RealT
        end

        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
        @test eltype(@inferred flux_godunov(u_ll, u_rr, normal_direction, equations)) ==
              RealT

        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT

        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LinearizedEulerEquations2D{Float32}
        @test eltype(adapted.v_mean_global) == Float32
        @test typeof(adapted.c_mean_global) == Float32
    end
end

@testitem "Type stability: Linearized Euler 3D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred LinearizedEulerEquations3D(v_mean_global = (RealT(0),
                                                                          RealT(0),
                                                                          RealT(0)),
                                                         c_mean_global = RealT(1),
                                                         rho_mean_global = RealT(1))

        x = SVector(zero(RealT), zero(RealT), zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT),
                                            one(RealT), one(RealT))
        orientations = [1, 2, 3]
        directions = [1, 2, 3, 4, 5, 6]
        normal_direction = SVector(one(RealT), zero(RealT), zero(RealT))

        surface_flux_function = flux_hll

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT

        for orientation in orientations
            for direction in directions
                @test eltype(@inferred boundary_condition_wall(u_inner, orientation,
                                                               direction, x, t,
                                                               surface_flux_function,
                                                               equations)) == RealT
            end

            @test eltype(@inferred flux(u, orientation, equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                       equations)) == RealT
        end

        @test eltype(@inferred flux(u, normal_direction, equations)) == RealT

        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT
        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                   equations)) == RealT

        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LinearizedEulerEquations3D{Float32}
        @test eltype(adapted.v_mean_global) == Float32
        @test typeof(adapted.c_mean_global) == Float32
    end
end

@testitem "Type stability: Polytropic Euler 2D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations1 = @inferred PolytropicEulerEquations2D(RealT(1),
                                                          RealT(1)) # equations.gamma == 1
        equations2 = @inferred PolytropicEulerEquations2D(RealT(1.4), RealT(0.5))  # equations.gamma > 1

        for equations in (equations1, equations2)
            x = SVector(zero(RealT), zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = SVector(one(RealT), one(RealT), one(RealT))
            orientations = [1, 2]
            directions = [1, 2, 3, 4]
            normal_direction = SVector(one(RealT), zero(RealT))

            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT
            @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
                  RealT
            @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
                  RealT

            @test eltype(@inferred velocity(u, normal_direction, equations)) == RealT
            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test eltype(@inferred flux_winters_etal(u_ll, u_rr, normal_direction,
                                                     equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                       equations)) ==
                  RealT
            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) ==
                  RealT

            for orientation in orientations
                @test eltype(@inferred velocity(u, orientation, equations)) == RealT
                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                @test eltype(@inferred flux_winters_etal(u_ll, u_rr, orientation,
                                                         equations)) ==
                      RealT
                @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                           equations)) ==
                      RealT
                @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) ==
                      RealT
            end

            @test eltype(@inferred velocity(u, equations)) == RealT
            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test typeof(@inferred density(u, equations)) == RealT
            @test typeof(@inferred pressure(u, equations)) == RealT
        end

        adapted1 = @inferred Trixi.trixi_adapt(Array, Float32, equations1)
        @test adapted1 isa PolytropicEulerEquations2D{Float32}
        @test typeof(adapted1.gamma) == Float32
        @test typeof(adapted1.kappa) == Float32
    end

    @timed_testset "Liu-Zhang positivity limiter" begin
        for RealT in (Float32, Float64)
            # ensure euler_arithmetic_tol < minimum(lower_bounds) in projection code
            rho_floor = RealT(1000) * eps(RealT)
            rho_e_floor = RealT(1000) * eps(RealT)
            lower_bounds = (rho_floor, rho_e_floor)
            variables = (density, energy_internal)

            # 1D compressible Euler
            equations_1d = CompressibleEulerEquations1D(RealT(5 / 3))

            converted_thresholds, converted_variables = @inferred Trixi.convert_variables_and_thresholds(lower_bounds,
                                                                                                         variables,
                                                                                                         equations_1d)
            @test eltype(converted_thresholds) == RealT
            @test converted_variables == (density, energy_internal)

            u_admissible = prim2cons(SVector(RealT(1), zero(RealT), RealT(1)), equations_1d)
            u_violation = SVector(rho_floor / 100, zero(RealT), RealT(1)) # violate density lower bound
            @test typeof(@inferred Trixi.state_is_admissible(u_admissible, lower_bounds,
                                                             variables, equations_1d)) ==
                  Bool
            @test typeof(@inferred Trixi.state_is_admissible(u_violation, lower_bounds,
                                                             variables, equations_1d)) ==
                  Bool

            @test eltype(@inferred Trixi.project_to_admissible_set(u_admissible,
                                                                   lower_bounds,
                                                                   variables, equations_1d)) ==
                  RealT
            @test eltype(@inferred Trixi.project_to_admissible_set(u_violation,
                                                                   lower_bounds,
                                                                   variables, equations_1d)) ==
                  RealT

            # 2D compressible Euler
            equations_2d = CompressibleEulerEquations2D(RealT(5 / 3))

            # no test for convert_variables_and_thresholds in 2D since it is the same as in 1D

            u_admissible = prim2cons(SVector(RealT(1), zero(RealT), zero(RealT), RealT(1)),
                                     equations_2d)
            u_violation = SVector(rho_floor / 100, zero(RealT), zero(RealT), RealT(1)) # violate density lower bound
            @test typeof(@inferred Trixi.state_is_admissible(u_admissible, lower_bounds,
                                                             variables, equations_2d)) ==
                  Bool
            @test typeof(@inferred Trixi.state_is_admissible(u_violation, lower_bounds,
                                                             variables, equations_2d)) ==
                  Bool

            @test eltype(@inferred Trixi.project_to_admissible_set(u_admissible,
                                                                   lower_bounds,
                                                                   variables, equations_2d)) ==
                  RealT
            @test eltype(@inferred Trixi.project_to_admissible_set(u_violation,
                                                                   lower_bounds,
                                                                   variables, equations_2d)) ==
                  RealT

            # check type of constructor and fields
            solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs, RealT = RealT)
            for (coordinates_min, coordinates_max, equations) in ((RealT(-1), RealT(1),
                                                                   equations_1d),
                                                                  ((RealT(-1), RealT(-1)),
                                                                   (RealT(1), RealT(1)),
                                                                   equations_2d))
                mesh = TreeMesh(coordinates_min, coordinates_max,
                                initial_refinement_level = 2, periodicity = true,
                                RealT = RealT)
                semi = SemidiscretizationHyperbolic(mesh, equations,
                                                    initial_condition_constant, solver;
                                                    boundary_conditions = boundary_condition_periodic)
                local_limiter! = PositivityPreservingLimiterZhangShu(;
                                                                     thresholds = lower_bounds,
                                                                     variables)
                global_limiter! = @inferred PositivityPreservingLimiterLiuZhang(local_limiter!,
                                                                                semi)

                nvars = nvariables(equations)
                SVectorT = SVector{nvars, RealT}
                for field in (:cell_averages, :davis_yin_dual_vars,
                              :projected_cell_averages)
                    @test eltype(getfield(global_limiter!, field)) == SVectorT
                end
                @test eltype(global_limiter!.sqrt_cell_volumes) == RealT
                @test typeof(global_limiter!.global_limiter_tol) == RealT
            end
        end
    end
end

@testitem "Type stability: Traffic Flow LWR 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        equations = @inferred TrafficFlowLWREquations1D(RealT(1))

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = SVector(one(RealT))
        orientation = 1
        c = one(RealT)

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT
        @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
              RealT

        @test eltype(@inferred flux(u, orientation, equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test typeof(@inferred entropy(c, equations)) == RealT
        @test typeof(@inferred entropy(u, equations)) == RealT
        @test typeof(@inferred energy_total(c, equations)) == RealT
        @test typeof(@inferred energy_total(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa TrafficFlowLWREquations1D{Float32}
        @test typeof(adapted.v_max) == Float32
    end
end

@testitem "Type stability: Passive tracer equations" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        # set gamma = 2 for the coupling convergence test
        flow_equations = @inferred CompressibleEulerEquations1D(RealT(2))
        equations = @inferred PassiveTracerEquations{1, 5, 2, typeof(flow_equations)}(flow_equations)

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = u_inner = cons = SVector(one(RealT), one(RealT), one(RealT),
                                                   one(RealT), one(RealT))
        orientation = 1
        directions = [1, 2]

        surface_flux_function = flux_lax_friedrichs
        @test eltype(@inferred initial_condition_density_wave(x, t, equations)) == RealT

        @test eltype(@inferred flux(u, orientation, equations)) == RealT

        flux_central = FluxTracerEquationsCentral(flux_ranocha)

        @test eltype(@inferred flux_central(u_ll, u_rr, orientation,
                                            equations)) ==
              RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT

        @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred Trixi.tracers(u, equations)) == RealT
        @test eltype(@inferred Trixi.rho_tracers(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT
        @test eltype(@inferred prim2cons(u, equations)) == RealT
        @test typeof(@inferred density(u, equations)) == RealT
        @test typeof(@inferred velocity(u, orientation, equations)) == RealT
        @test typeof(@inferred pressure(u, equations)) == RealT
        @test typeof(@inferred density_pressure(u, equations)) == RealT
        @test typeof(@inferred entropy(cons, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa PassiveTracerEquations
        @test adapted.flow_equations isa CompressibleEulerEquations1D{Float32}
    end
end

@testitem "Type stability: Linear Elasticity 1D" setup=[Setup, TypeStability] tags=[:misc_part1] begin
    for RealT in (Float32, Float64)
        rho = RealT(42)
        mu = RealT(1)
        lambda = RealT(7)
        equations = @inferred LinearElasticityEquations1D(rho = rho,
                                                          mu = mu,
                                                          lambda = lambda)

        x = SVector(zero(RealT))
        t = zero(RealT)
        u = u_ll = u_rr = SVector(one(RealT), zero(RealT))
        orientation = 1

        @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
              RealT

        @test eltype(@inferred flux(u, orientation, equations)) == RealT

        @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation, equations)) ==
              RealT
        @test eltype(@inferred Trixi.max_abs_speeds(equations)) == RealT

        @test eltype(@inferred cons2prim(u, equations)) == RealT
        @test eltype(@inferred cons2entropy(u, equations)) == RealT

        @test typeof(@inferred entropy(u, equations)) == RealT

        @test typeof(@inferred energy_total(u, equations)) == RealT
        @test typeof(@inferred energy_internal(u, equations)) == RealT
        @test typeof(@inferred energy_kinetic(u, equations)) == RealT

        @test typeof(@inferred velocity(u, equations)) == RealT

        adapted = @inferred Trixi.trixi_adapt(Array, Float32, equations)
        @test adapted isa LinearElasticityEquations1D{Float32}
        @test typeof(adapted.rho) == Float32
        @test typeof(adapted.c1) == Float32
        @test typeof(adapted.E) == Float32
    end
end
