module TestType

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

# Run unit tests for various equations
@testset "Test Type Stability" begin
    @timed_testset "Acoustic Perturbation 2D" begin
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
        end
    end

    @timed_testset "Compressible Euler 1D" begin
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
            if RealT == Float32
                # check `ln_mean` (test broken)
                @test_broken eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                                 equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                          equations)) ==
                      RealT
            end
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_ranocha(u_ll, u_rr, orientation,
                                                           equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                      RealT
            end

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
            @test typeof(@inferred pressure(u, equations)) == RealT
            @test typeof(@inferred density_pressure(u, equations)) == RealT
            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
        end
    end

    @timed_testset "Compressible Euler 2D" begin
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
            if RealT == Float32
                # check `ln_mean` (test broken)
                @test_broken eltype(@inferred flux_chandrashekar(u_ll, u_rr,
                                                                 normal_direction,
                                                                 equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, normal_direction,
                                                          equations)) ==
                      RealT
            end
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                                           equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                                    equations)) == RealT
            end

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
                if RealT == Float32
                    # check `ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_chandrashekar(u_ll, u_rr,
                                                                     orientation,
                                                                     equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                              equations)) ==
                          RealT
                end
                if RealT == Float32
                    # check `ln_mean` and `inv_ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_ranocha(u_ll, u_rr, orientation,
                                                               equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                          RealT
                end

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

            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test eltype(@inferred entropy2cons(u, equations)) == RealT
            @test eltype(@inferred Trixi.cons2entropy_guermond_etal(u, equations)) == RealT
            @test typeof(@inferred Trixi.entropy_guermond_etal(u, equations)) == RealT
            @test typeof(@inferred density(u, equations)) == RealT
            @test typeof(@inferred pressure(u, equations)) == RealT
            @test typeof(@inferred density_pressure(u, equations)) == RealT
            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred Trixi.entropy_math(cons, equations)) == RealT
            @test typeof(@inferred Trixi.entropy_thermodynamic(cons, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT

            # TODO: test `gradient_conservative`, not necessary but good to have
        end
    end

    @timed_testset "Compressible Euler 3D" begin
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

            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test eltype(@inferred flux_shima_etal(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred flux_kennedy_gruber(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred flux_lmars(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred flux_hllc(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            if RealT == Float32
                # check `ln_mean` (test broken)
                @test_broken eltype(@inferred flux_chandrashekar(u_ll, u_rr,
                                                                 normal_direction,
                                                                 equations)) == RealT
            else
                @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, normal_direction,
                                                          equations)) == RealT
            end
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                                           equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                                    equations)) == RealT
            end

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
                if RealT == Float32
                    # check `ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_chandrashekar(u_ll, u_rr,
                                                                     orientation,
                                                                     equations)) == RealT
                else
                    @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                              equations)) == RealT
                end
                if RealT == Float32
                    # check `ln_mean` and `inv_ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_ranocha(u_ll, u_rr, orientation,
                                                               equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                          RealT
                end

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

            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test eltype(@inferred entropy2cons(u, equations)) == RealT
            @test typeof(@inferred density(u, equations)) == RealT
            @test typeof(@inferred pressure(u, equations)) == RealT
            @test typeof(@inferred density_pressure(u, equations)) == RealT
            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred Trixi.entropy_math(cons, equations)) == RealT
            @test typeof(@inferred Trixi.entropy_thermodynamic(cons, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
        end
    end

    @timed_testset "Compressible Euler Multicomponent 1D" begin
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

            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            if RealT == Float32
                # check `ln_mean` (test broken)
                @test_broken eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                                 equations)) == RealT
            else
                @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                          equations)) == RealT
            end
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_ranocha(u_ll, u_rr, orientation,
                                                           equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                      RealT
            end

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
        end
    end

    @timed_testset "Compressible Euler Multicomponent 2D" begin
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

            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                                           equations)) == RealT
            else
                @test eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                                    equations)) == RealT
            end

            for orientation in orientations
                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                if RealT == Float32
                    # check `ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_chandrashekar(u_ll, u_rr,
                                                                     orientation,
                                                                     equations)) == RealT
                else
                    @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                              equations)) == RealT
                end
                if RealT == Float32
                    # check `ln_mean` and `inv_ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_ranocha(u_ll, u_rr, orientation,
                                                               equations)) == RealT
                else
                    @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                          RealT
                end

                @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) ==
                      RealT
            end

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
        end
    end

    @timed_testset "Compressible Euler Quasi 1D" begin
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
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_chan_etal(u_ll, u_rr, orientation,
                                                             equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_chan_etal(u_ll, u_rr, orientation, equations)) ==
                      RealT
            end

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
        end
    end

    @timed_testset "Compressible Navier Stokes Diffusion 1D" begin
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

            u = u_transformed = SVector(one(RealT), zero(RealT),
                                        zero(RealT))
            orientation = 1
            gradients = SVector(RealT(0.1), RealT(0.1), RealT(0.1))

            for equations_parabolic in (equations_parabolic_primitive,
                                        equations_parabolic_entropy)
                @test eltype(@inferred flux(u, gradients, orientation, equations_parabolic)) ==
                      RealT

                @test eltype(@inferred cons2prim(u, equations_parabolic)) == RealT
                @test eltype(@inferred prim2cons(u, equations_parabolic)) == RealT
                @test eltype(@inferred cons2entropy(u, equations_parabolic)) == RealT
                @test eltype(@inferred entropy2cons(u, equations_parabolic)) == RealT
                @test typeof(@inferred Trixi.temperature(u, equations_parabolic)) == RealT

                @test eltype(@inferred Trixi.convert_transformed_to_primitive(u_transformed,
                                                                              equations_parabolic)) ==
                      RealT
                @test eltype(@inferred Trixi.convert_derivative_to_primitive(u, gradients,
                                                                             equations_parabolic)) ==
                      RealT
            end

            # TODO: BC tests for GradientVariablesPrimitive
            # TODO: BC tests for GradientVariablesEntropy
        end
    end

    @timed_testset "Compressible Navier Stokes Diffusion 2D" begin
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

            u = u_transformed = SVector(one(RealT), zero(RealT), zero(RealT), zero(RealT))
            orientations = [1, 2]
            gradient = SVector(RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1))
            gradients = SVector(gradient, gradient)

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
                @test typeof(@inferred Trixi.enstrophy(u, gradients, equations_parabolic)) ==
                      RealT
                @test typeof(@inferred Trixi.vorticity(u, gradients, equations_parabolic)) ==
                      RealT

                @test eltype(@inferred Trixi.convert_transformed_to_primitive(u_transformed,
                                                                              equations_parabolic)) ==
                      RealT
                @test eltype(@inferred Trixi.convert_derivative_to_primitive(u, gradient,
                                                                             equations_parabolic)) ==
                      RealT
            end

            # TODO: BC tests for GradientVariablesPrimitive
            # TODO: BC tests for GradientVariablesEntropy
        end
    end

    @timed_testset "Compressible Navier Stokes Diffusion 3D" begin
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

            u = u_transformed = SVector(one(RealT), zero(RealT), zero(RealT), zero(RealT),
                                        zero(RealT))
            orientations = [1, 2, 3]
            gradient = SVector(RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1), RealT(0.1))
            gradients = SVector(gradient, gradient, gradient)

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
                @test typeof(@inferred Trixi.enstrophy(u, gradients, equations_parabolic)) ==
                      RealT
                @test eltype(@inferred Trixi.vorticity(u, gradients, equations_parabolic)) ==
                      RealT

                @test eltype(@inferred Trixi.convert_transformed_to_primitive(u_transformed,
                                                                              equations_parabolic)) ==
                      RealT
                @test eltype(@inferred Trixi.convert_derivative_to_primitive(u, gradient,
                                                                             equations_parabolic)) ==
                      RealT
            end

            # TODO: BC tests for GradientVariablesPrimitive
            # TODO: BC tests for GradientVariablesEntropy
        end
    end

    @timed_testset "Hyperbolic Diffusion 1D" begin
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
        end
    end

    @timed_testset "Hyperbolic Diffusion 2D" begin
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
        end
    end

    @timed_testset "Hyperbolic Diffusion 3D" begin
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
        end
    end

    @timed_testset "Ideal Glm Mhd 1D" begin
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

            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_hllc(u_ll, u_rr, orientation, equations)) == RealT
            if RealT == Float32
                # check `ln_mean` (test broken)
                @test_broken eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                               equations)) == RealT
                # check `ln_mean` and `inv_ln_mean` (test broken) 
                @test_broken eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                                      orientation,
                                                                      equations)) == RealT
            else
                @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation, equations)) ==
                      RealT
                @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                               orientation,
                                                               equations)) == RealT
            end

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

            @test typeof(@inferred Trixi.entropy_thermodynamic(cons, equations)) == RealT
            @test typeof(@inferred Trixi.entropy_math(cons, equations)) == RealT
            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_total(cons, equations)) == RealT
            @test typeof(@inferred energy_kinetic(cons, equations)) == RealT
            @test typeof(@inferred energy_magnetic(cons, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
            @test typeof(@inferred cross_helicity(cons, equations)) == RealT
        end
    end

    @timed_testset "Ideal Glm Mhd 2D" begin
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
            normal_direction = normal_direction_ll = normal_direction_average = SVector(one(RealT),
                                                                                        zero(RealT))
            nonconservative_type_local = Trixi.NonConservativeLocal()
            nonconservative_type_symmetric = Trixi.NonConservativeSymmetric()
            nonconservative_terms = [1, 2]

            @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT
            @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
                  RealT

            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr,
                                                               normal_direction_ll,
                                                               normal_direction_average,
                                                               equations)) == RealT
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                                      normal_direction,
                                                                      equations)) == RealT
            else
                @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, normal_direction,
                                                               equations)) == RealT
            end

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, normal_direction,
                                                          equations)) == RealT

            for orientation in orientations
                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr, orientation,
                                                                   equations)) == RealT
                @test eltype(@inferred flux_nonconservative_powell_local_symmetric(u_ll,
                                                                                   u_rr,
                                                                                   orientation,
                                                                                   equations)) ==
                      RealT
                if RealT == Float32
                    # check `ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                                   equations)) ==
                                 RealT
                    # check `ln_mean` and `inv_ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                                          orientation,
                                                                          equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                            equations)) == RealT
                    @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, orientation,
                                                                   equations)) == RealT
                end
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

            @test typeof(@inferred Trixi.entropy_thermodynamic(cons, equations)) == RealT
            @test typeof(@inferred Trixi.entropy_math(cons, equations)) == RealT
            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_total(cons, equations)) == RealT
            @test typeof(@inferred energy_kinetic(cons, equations)) == RealT
            @test typeof(@inferred energy_magnetic(cons, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
            @test typeof(@inferred cross_helicity(cons, equations)) == RealT
        end
    end

    @timed_testset "Ideal Glm Mhd 3D" begin
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
            normal_direction = normal_direction_ll = normal_direction_average = SVector(one(RealT),
                                                                                        zero(RealT),
                                                                                        zero(RealT))

            @test eltype(@inferred initial_condition_constant(x, t, equations)) == RealT
            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT
            @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
                  RealT

            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr,
                                                               normal_direction_ll,
                                                               normal_direction_average,
                                                               equations)) == RealT
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                                      normal_direction,
                                                                      equations)) == RealT
            else
                @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, normal_direction,
                                                               equations)) == RealT
            end

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, normal_direction,
                                                          equations)) == RealT

            for orientation in orientations
                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                @test eltype(@inferred flux_nonconservative_powell(u_ll, u_rr, orientation,
                                                                   equations)) == RealT
                if RealT == Float32
                    # check `ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                                   equations)) ==
                                 RealT
                    # check `ln_mean` and `inv_ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                                          orientation,
                                                                          equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                            equations)) == RealT
                    @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, orientation,
                                                                   equations)) == RealT
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

            @test typeof(@inferred Trixi.entropy_thermodynamic(cons, equations)) == RealT
            @test typeof(@inferred Trixi.entropy_math(cons, equations)) == RealT
            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_total(cons, equations)) == RealT
            @test typeof(@inferred energy_kinetic(cons, equations)) == RealT
            @test typeof(@inferred energy_magnetic(cons, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
            @test typeof(@inferred cross_helicity(cons, equations)) == RealT
        end
    end

    @timed_testset "Ideal Glm Mhd Multicomponent 1D" begin
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
            if RealT == Float32
                # check `ln_mean` (test broken)
                @test_broken eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                               equations)) ==
                             RealT
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                                      orientation,
                                                                      equations)) == RealT
            else
                @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation, equations)) ==
                      RealT
                @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, orientation,
                                                               equations)) == RealT
            end

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
        end
    end

    @timed_testset "Ideal Glm Mhd Multicomponent 2D" begin
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
                if RealT == Float32
                    # check `ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                                   equations)) == RealT
                    # check `ln_mean` and `inv_ln_mean` (test broken)
                    @test_broken eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr,
                                                                          orientation,
                                                                          equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_derigs_etal(u_ll, u_rr, orientation,
                                                            equations)) ==
                          RealT
                    @test eltype(@inferred flux_hindenlang_gassner(u_ll, u_rr, orientation,
                                                                   equations)) == RealT
                end

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
        end
    end

    @timed_testset "Inviscid Burgers 1D" begin
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

    @timed_testset "Lattice Boltzmann 2D" begin
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
        end
    end

    @timed_testset "Lattice Boltzmann 3D" begin
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
        end
    end

    @timed_testset "Linear Scalar Advection 1D" begin
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
        end
    end

    @timed_testset "Linear Scalar Advection 2D" begin
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
        end
    end

    @timed_testset "Linear Scalar Advection 3D" begin
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
        end
    end

    @timed_testset "Maxwell 1D" begin
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
        end
    end

    @timed_testset "Linearized Euler 1D" begin
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
        end
    end

    @timed_testset "Linearized Euler 2D" begin
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
        end
    end

    @timed_testset "Linearized Euler 3D" begin
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
        end
    end

    @timed_testset "Polytropic Euler 2D" begin
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

                @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
                if RealT == Float32
                    # check `ln_mean` and `stolarsky_mean` (test broken)
                    @test_broken eltype(@inferred flux_winters_etal(u_ll, u_rr,
                                                                    normal_direction,
                                                                    equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_winters_etal(u_ll, u_rr, normal_direction,
                                                             equations)) ==
                          RealT
                end
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
                    @test eltype(@inferred flux(u, orientation, equations)) == RealT
                    if RealT == Float32
                        # check `ln_mean` and `stolarsky_mean` (test broken)
                        @test_broken eltype(@inferred flux_winters_etal(u_ll, u_rr,
                                                                        orientation,
                                                                        equations)) ==
                                     RealT
                    else
                        @test eltype(@inferred flux_winters_etal(u_ll, u_rr, orientation,
                                                                 equations)) ==
                              RealT
                    end
                    @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                               equations)) ==
                          RealT
                    @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                               equations)) ==
                          RealT
                end

                @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
                @test eltype(@inferred cons2prim(u, equations)) == RealT
                @test eltype(@inferred prim2cons(u, equations)) == RealT
                @test eltype(@inferred cons2entropy(u, equations)) == RealT
                @test typeof(@inferred density(u, equations)) == RealT
                @test typeof(@inferred pressure(u, equations)) == RealT
            end
        end
    end

    @timed_testset "Shallow Water 1D" begin
        for RealT in (Float32, Float64)
            equations = @inferred ShallowWaterEquations1D(gravity_constant = RealT(9.81))

            x = SVector(zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = u_inner = cons = SVector(one(RealT), one(RealT), one(RealT))
            orientation = 1
            directions = [1, 2]
            normal_direction = SVector(one(RealT))

            surface_flux_function = flux_lax_friedrichs
            dissipation = DissipationLocalLaxFriedrichs()
            numflux = FluxHLL()

            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT
            @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
                  RealT
            @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
                  RealT

            for direction in directions
                @test eltype(@inferred boundary_condition_slip_wall(u_inner,
                                                                    orientation,
                                                                    direction,
                                                                    x, t,
                                                                    surface_flux_function,
                                                                    equations)) == RealT
                @test eltype(@inferred Trixi.calc_wavespeed_roe(u_ll, u_rr, direction,
                                                                equations)) ==
                      RealT
            end

            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                                         orientation,
                                                                         equations)) ==
                  RealT
            @test eltype(@inferred flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                                                       orientation,
                                                                       equations)) == RealT
            @test eltype(@inferred flux_nonconservative_audusse_etal(u_ll, u_rr,
                                                                     orientation,
                                                                     equations)) == RealT
            @test eltype(@inferred flux_nonconservative_ersing_etal(u_ll, u_rr, orientation,
                                                                    equations)) == RealT
            @test eltype(@inferred flux_fjordholm_etal(u_ll, u_rr, orientation,
                                                       equations)) == RealT
            @test eltype(@inferred flux_wintermeyer_etal(u_ll, u_rr, orientation,
                                                         equations)) == RealT

            @test eltype(eltype(@inferred hydrostatic_reconstruction_audusse_etal(u_ll,
                                                                                  u_rr,
                                                                                  equations))) ==
                  RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred dissipation(u_ll, u_rr, orientation, equations)) == RealT
            @test eltype(@inferred dissipation(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred numflux(u_ll, u_rr, orientation, equations)) == RealT
            # no matching method 
            # @test eltype(@inferred numflux(u_ll, u_rr, normal_direction, equations)) == RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                          equations)) == RealT
            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT

            @test typeof(@inferred velocity(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test eltype(@inferred entropy2cons(u, equations)) == RealT
            @test typeof(@inferred Trixi.waterheight(u, equations)) == RealT
            @test typeof(@inferred pressure(u, equations)) == RealT
            @test typeof(@inferred waterheight_pressure(u, equations)) == RealT

            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_total(cons, equations)) == RealT
            @test typeof(@inferred energy_kinetic(u, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
            @test typeof(@inferred lake_at_rest_error(u, equations)) == RealT
        end
    end

    @timed_testset "Shallow Water 2D" begin
        for RealT in (Float32, Float64)
            equations = @inferred ShallowWaterEquations2D(gravity_constant = RealT(9.81))

            x = SVector(zero(RealT), zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = u_inner = cons = SVector(one(RealT), one(RealT), one(RealT),
                                                       one(RealT))
            orientations = [1, 2]
            directions = [1, 2, 3, 4]
            normal_direction = normal_direction_ll = normal_direction_average = SVector(one(RealT),
                                                                                        zero(RealT))

            surface_flux_function = flux_lax_friedrichs
            dissipation = DissipationLocalLaxFriedrichs()
            numflux = FluxHLL()

            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT
            @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
                  RealT
            @test eltype(@inferred initial_condition_weak_blast_wave(x, t, equations)) ==
                  RealT
            @test eltype(@inferred boundary_condition_slip_wall(u_inner,
                                                                normal_direction,
                                                                x, t,
                                                                surface_flux_function,
                                                                equations)) == RealT

            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                                         normal_direction_ll,
                                                                         normal_direction_average,
                                                                         equations)) ==
                  RealT
            @test eltype(@inferred flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                                                       normal_direction_ll,
                                                                       normal_direction_average,
                                                                       equations)) == RealT
            @test eltype(@inferred flux_nonconservative_audusse_etal(u_ll, u_rr,
                                                                     normal_direction_ll,
                                                                     normal_direction_average,
                                                                     equations)) == RealT
            @test eltype(@inferred flux_nonconservative_ersing_etal(u_ll, u_rr,
                                                                    normal_direction_ll,
                                                                    normal_direction_average,
                                                                    equations)) == RealT
            @test eltype(@inferred flux_fjordholm_etal(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred flux_wintermeyer_etal(u_ll, u_rr, normal_direction,
                                                         equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred dissipation(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred numflux(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, normal_direction,
                                                          equations)) == RealT
            @test eltype(@inferred Trixi.calc_wavespeed_roe(u_ll, u_rr, normal_direction,
                                                            equations)) == RealT

            for orientation in orientations
                for direction in directions
                    @test eltype(@inferred boundary_condition_slip_wall(u_inner,
                                                                        orientation,
                                                                        direction, x, t,
                                                                        surface_flux_function,
                                                                        equations)) ==
                          RealT
                end

                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                @test eltype(@inferred flux_nonconservative_wintermeyer_etal(u_ll, u_rr,
                                                                             orientation,
                                                                             equations)) ==
                      RealT
                @test eltype(@inferred flux_nonconservative_fjordholm_etal(u_ll, u_rr,
                                                                           orientation,
                                                                           equations)) ==
                      RealT
                @test eltype(@inferred flux_nonconservative_audusse_etal(u_ll, u_rr,
                                                                         orientation,
                                                                         equations)) ==
                      RealT
                @test eltype(eltype(@inferred flux_nonconservative_ersing_etal(u_ll, u_rr,
                                                                               orientation,
                                                                               equations))) ==
                      RealT
                @test eltype(@inferred flux_fjordholm_etal(u_ll, u_rr, orientation,
                                                           equations)) == RealT
                @test eltype(@inferred flux_wintermeyer_etal(u_ll, u_rr, orientation,
                                                             equations)) == RealT

                @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) ==
                      RealT
                @test eltype(@inferred dissipation(u_ll, u_rr, orientation, equations)) ==
                      RealT
                @test eltype(@inferred numflux(u_ll, u_rr, orientation, equations)) == RealT
                @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) == RealT
                @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                           equations)) == RealT
                @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                              equations)) == RealT
                @test eltype(@inferred Trixi.calc_wavespeed_roe(u_ll, u_rr, orientation,
                                                                equations)) == RealT
            end

            @test eltype(eltype(@inferred hydrostatic_reconstruction_audusse_etal(u_ll,
                                                                                  u_rr,
                                                                                  equations))) ==
                  RealT

            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred velocity(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test eltype(@inferred entropy2cons(u, equations)) == RealT
            @test typeof(@inferred Trixi.waterheight(u, equations)) == RealT
            @test typeof(@inferred pressure(u, equations)) == RealT
            @test typeof(@inferred waterheight_pressure(u, equations)) == RealT

            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_total(cons, equations)) == RealT
            @test typeof(@inferred energy_kinetic(u, equations)) == RealT
            @test typeof(@inferred energy_internal(cons, equations)) == RealT
            @test typeof(@inferred lake_at_rest_error(u, equations)) == RealT
        end
    end

    @timed_testset "Shallow Water Quasi 1D" begin
        for RealT in (Float32, Float64)
            equations = @inferred ShallowWaterEquationsQuasi1D(gravity_constant = RealT(9.81))

            x = SVector(zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = cons = SVector(one(RealT), one(RealT), one(RealT), one(RealT))
            orientation = 1
            normal_direction = normal_ll = normal_rr = SVector(one(RealT))

            dissipation = DissipationLocalLaxFriedrichs()

            @test eltype(@inferred initial_condition_convergence_test(x, t, equations)) ==
                  RealT
            @test eltype(@inferred source_terms_convergence_test(u, x, t, equations)) ==
                  RealT

            @test eltype(@inferred flux(u, orientation, equations)) == RealT
            @test eltype(@inferred flux_nonconservative_chan_etal(u_ll, u_rr,
                                                                  orientation,
                                                                  equations)) ==
                  RealT
            @test eltype(@inferred flux_nonconservative_chan_etal(u_ll, u_rr,
                                                                  normal_direction,
                                                                  equations)) ==
                  RealT
            @test eltype(@inferred flux_nonconservative_chan_etal(u_ll, u_rr, normal_ll,
                                                                  normal_rr,
                                                                  equations)) == RealT
            @test eltype(@inferred flux_chan_etal(u_ll, u_rr, orientation,
                                                  equations)) == RealT
            @test eltype(@inferred flux_chan_etal(u_ll, u_rr, normal_direction,
                                                  equations)) == RealT

            @test typeof(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred dissipation(u_ll, u_rr, orientation, equations)) == RealT
            @test eltype(@inferred dissipation(u_ll, u_rr, normal_direction, equations)) ==
                  RealT
            @test eltype(@inferred Trixi.max_abs_speeds(u, equations)) == RealT
            @test typeof(@inferred velocity(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test typeof(@inferred Trixi.waterheight(u, equations)) == RealT

            @test typeof(@inferred entropy(cons, equations)) == RealT
            @test typeof(@inferred energy_total(cons, equations)) == RealT
            @test typeof(@inferred lake_at_rest_error(u, equations)) == RealT
        end
    end

    @timed_testset "Traffic Flow LWR 1D" begin
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
        end
    end
end

end # module
