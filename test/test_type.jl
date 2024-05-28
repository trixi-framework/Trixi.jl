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
                    if RealT == Float32
                        # check `surface_flux_function` (test broken)
                        @test_broken eltype(boundary_condition_wall(u_inner, orientation,
                                                                    direction, x, t,
                                                                    surface_flux_function,
                                                                    equations)) == RealT
                    else
                        @test eltype(@inferred boundary_condition_wall(u_inner, orientation,
                                                                       direction, x, t,
                                                                       surface_flux_function,
                                                                       equations)) == RealT
                    end
                end
            end

            if RealT == Float32
                # check `surface_flux_function` (test broken)
                @test_broken eltype(boundary_condition_slip_wall(u_inner, normal_direction,
                                                                 x, t,
                                                                 surface_flux_function,
                                                                 equations)) == RealT
            else
                @test eltype(@inferred boundary_condition_slip_wall(u_inner,
                                                                    normal_direction, x, t,
                                                                    surface_flux_function,
                                                                    equations)) ==
                      RealT
            end

            @test eltype(@inferred flux(u, normal_direction, equations)) == RealT
            @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
                                                       equations)) ==
                  RealT
            @test eltype(@inferred dissipation(u_ll, u_rr, normal_direction, equations)) ==
                  RealT

            for orientation in orientations
                @test eltype(@inferred flux(u, orientation, equations)) == RealT
                @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) ==
                      RealT
                @test eltype(@inferred dissipation(u_ll, u_rr, orientation, equations)) ==
                      RealT
            end

            @test eltype(@inferred max_abs_speeds(u, equations)) == RealT
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
            u = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT))
            orientation = 1
            directions = [1, 2]
            cons = SVector(one(RealT), one(RealT), one(RealT))

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
                @test_broken eltype(flux_chandrashekar(u_ll, u_rr, orientation, equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                          equations)) ==
                      RealT
            end
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(flux_ranocha(u_ll, u_rr, orientation, equations)) ==
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

            @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation, equations)) ==
                  RealT
            @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                          equations)) ==
                  RealT

            @test eltype(@inferred max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test eltype(@inferred entropy2cons(u, equations)) == RealT
            @test eltype(@inferred density(u, equations)) == RealT
            @test eltype(@inferred pressure(u, equations)) == RealT
            @test eltype(@inferred density_pressure(u, equations)) == RealT
            @test eltype(@inferred entropy(cons, equations)) == RealT
            @test eltype(@inferred energy_internal(cons, equations)) == RealT
        end
    end

    @timed_testset "Compressible Euler 2D" begin
        for RealT in (Float32, Float64)
            # set gamma = 2 for the coupling convergence test\
            equations = @inferred CompressibleEulerEquations2D(RealT(2))

            x = SVector(zero(RealT), zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT),
                                                one(RealT))
            orientations = [1, 2]
            directions = [1, 2, 3, 4]
            normal_direction = SVector(one(RealT), zero(RealT))
            cons = SVector(one(RealT), one(RealT), one(RealT), one(RealT))

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
                @test_broken eltype(flux_chandrashekar(u_ll, u_rr, normal_direction,
                                                       equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, normal_direction,
                                                          equations)) ==
                      RealT
            end
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(flux_ranocha(u_ll, u_rr, normal_direction, equations)) ==
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

            @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
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
                    @test_broken eltype(flux_chandrashekar(u_ll, u_rr, orientation,
                                                           equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                              equations)) ==
                          RealT
                end
                if RealT == Float32
                    # check `ln_mean` and `inv_ln_mean` (test broken)
                    @test_broken eltype(flux_ranocha(u_ll, u_rr, orientation, equations)) ==
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

                @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
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

            @test eltype(@inferred max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test eltype(@inferred entropy2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy_guermond_etal(u, equations)) == RealT
            @test eltype(@inferred density(u, equations)) == RealT
            @test eltype(@inferred pressure(u, equations)) == RealT
            @test eltype(@inferred density_pressure(u, equations)) == RealT
            @test eltype(@inferred entropy(cons, equations)) == RealT
            @test eltype(@inferred entropy_math(cons, equations)) == RealT
            @test eltype(@inferred entropy_thermodynamic(cons, equations)) == RealT
            @test eltype(@inferred energy_internal(cons, equations)) == RealT
            # TODO: test `gradient_conservative`, not necessary but good to have
        end
    end

    @timed_testset "Compressible Euler 3D" begin
        for RealT in (Float32, Float64)
            # set gamma = 2 for the coupling convergence test 
            equations = @inferred CompressibleEulerEquations3D(RealT(2))

            x = SVector(zero(RealT), zero(RealT), zero(RealT))
            t = zero(RealT)
            u = u_ll = u_rr = u_inner = SVector(one(RealT), one(RealT), one(RealT),
                                                one(RealT), one(RealT))
            orientations = [1, 2, 3]
            directions = [1, 2, 3, 4, 5, 6]
            normal_direction = SVector(one(RealT), zero(RealT), zero(RealT))
            cons = SVector(one(RealT), one(RealT), one(RealT),
                           one(RealT), one(RealT))

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
                @test_broken eltype(flux_chandrashekar(u_ll, u_rr, normal_direction,
                                                       equations)) == RealT
            else
                @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, normal_direction,
                                                          equations)) == RealT
            end
            if RealT == Float32
                # check `ln_mean` and `inv_ln_mean` (test broken)
                @test_broken eltype(flux_ranocha(u_ll, u_rr, normal_direction, equations)) ==
                             RealT
            else
                @test eltype(@inferred flux_ranocha(u_ll, u_rr, normal_direction,
                                                    equations)) == RealT
            end

            @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, normal_direction,
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
                    @test_broken eltype(flux_chandrashekar(u_ll, u_rr, orientation,
                                                           equations)) == RealT
                else
                    @test eltype(@inferred flux_chandrashekar(u_ll, u_rr, orientation,
                                                              equations)) == RealT
                end
                if RealT == Float32
                    # check `ln_mean` and `inv_ln_mean` (test broken)
                    @test_broken eltype(flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                                 RealT
                else
                    @test eltype(@inferred flux_ranocha(u_ll, u_rr, orientation, equations)) ==
                          RealT
                end

                @test eltype(eltype(@inferred splitting_steger_warming(u, orientation,
                                                                       equations))) ==
                      RealT

                @test eltype(@inferred max_abs_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) == RealT
                @test eltype(@inferred min_max_speed_naive(u_ll, u_rr, orientation,
                                                           equations)) == RealT
                @test eltype(@inferred min_max_speed_davis(u_ll, u_rr, orientation,
                                                           equations)) == RealT
                @test eltype(@inferred min_max_speed_einfeldt(u_ll, u_rr, orientation,
                                                              equations)) == RealT
            end

            @test eltype(@inferred max_abs_speeds(u, equations)) == RealT
            @test eltype(@inferred cons2prim(u, equations)) == RealT
            @test eltype(@inferred prim2cons(u, equations)) == RealT
            @test eltype(@inferred cons2entropy(u, equations)) == RealT
            @test eltype(@inferred entropy2cons(u, equations)) == RealT
            @test eltype(@inferred density(u, equations)) == RealT
            @test eltype(@inferred pressure(u, equations)) == RealT
            @test eltype(@inferred density_pressure(u, equations)) == RealT
            @test eltype(@inferred entropy(cons, equations)) == RealT
            @test eltype(@inferred entropy_math(cons, equations)) == RealT
            @test eltype(@inferred entropy_thermodynamic(cons, equations)) == RealT
            @test eltype(@inferred energy_internal(cons, equations)) == RealT
        end
    end
end

end # module
