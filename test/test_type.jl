module TestType

using Test
using Trixi

include("test_trixi.jl")

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

# Run unit tests for various equations
@testset "Float32 Type Stability" begin
    @timed_testset "Acoustic Perturbation 2D" begin
        v_mean_global = (0.0f0, 0.0f0)
        c_mean_global = 1.0f0
        rho_mean_global = 1.0f0
        equations = AcousticPerturbationEquations2D(v_mean_global, c_mean_global,
                                                    rho_mean_global)

        x = SVector(0.0f0, 0.0f0)
        t = 0.0f0
        u = u_ll = u_rr = SVector(1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0)
        orientations = [1, 2]
        normal_direction = SVector(1.0f0, 1.0f0)

        @test eltype(initial_condition_constant(x, t, equations)) == Float32
        @test eltype(initial_condition_convergence_test(x, t, equations)) == Float32
        @test eltype(initial_condition_gauss(x, t, equations)) == Float32

        @test eltype(source_terms_convergence_test(u, x, t, equations)) == Float32
        @test eltype(source_terms_convergence_test(u, x, t, equations)) == Float32

        for orientation in orientations
            @test eltype(flux(u, orientation, equations)) == Float32
        end

        @test eltype(flux(u, normal_direction, equations)) == Float32

        dissipation = DissipationLocalLaxFriedrichs()
        # test orientation
        @test eltype(dissipation(u_ll, u_rr, orientations[1], equations)) == Float32
        # test normal direction
        @test eltype(dissipation(u_ll, u_rr, normal_direction, equations)) == Float32
    end

    @timed_testset "Compressible Euler 1D" begin
        # set to 2.0f0 for coupling convergence test
        equations = CompressibleEulerEquations1D(2.0f0)

        x = SVector(0.0f0)
        t = 0.0f0
        u = u_ll = u_rr = u_inner = SVector(1.0f0, 1.0f0, 1.0f0)
        orientation = 1
        directions = [1, 2] # even and odd

        @test eltype(initial_condition_constant(x, t, equations)) == Float32
        @test eltype(initial_condition_convergence_test(x, t, equations)) == Float32
        @test eltype(initial_condition_density_wave(x, t, equations)) == Float32
        @test eltype(initial_condition_weak_blast_wave(x, t, equations)) == Float32
        @test eltype(initial_condition_eoc_test_coupled_euler_gravity(x, t, equations)) ==
              Float32

        @test eltype(source_terms_convergence_test(u, x, t, equations)) == Float32

        # set a surface flux function as ?
        #= for direction in directions 
            @test eltype(boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                                      surface_flux_function, equations)) == Float32
        end =#

        @test eltype(flux(u, orientation, equations)) == Float32
        @test eltype(flux_shima_etal(u_ll, u_rr, orientation, equations)) == Float32
        @test eltype(flux_kennedy_gruber(u_ll, u_rr, orientation, equations)) == Float32
        @test_broken eltype(flux_chandrashekar(u_ll, u_rr, orientation, equations)) ==
                     Float32 # ln_mean
        @test_broken eltype(flux_ranocha(u_ll, u_rr, orientation, equations)) == Float32 # ln_mean, inv_ln_mean
        @test eltype(flux_hllc(u_ll, u_rr, orientation, equations)) == Float32

        @test eltype(eltype(splitting_steger_warming(u, orientation, equations))) == Float32
        @test eltype(eltype(splitting_vanleer_haenel(u, orientation, equations))) == Float32
        @test eltype(eltype(splitting_coirier_vanleer(u, orientation, equations))) ==
              Float32

        @test eltype(max_abs_speed_naive(u_ll, u_rr, orientation, equations)) == Float32
        @test eltype(min_max_speed_einfeldt(u_ll, u_rr, orientation, equations)) == Float32
        # @test eltype(max_abs_speeds(u, equations)) == Float32 # not defined ?

        @test eltype(cons2prim(u, equations)) == Float32
        @test eltype(prim2cons(u, equations)) == Float32
        @test eltype(cons2entropy(u, equations)) == Float32
        @test eltype(entropy2cons(u, equations)) == Float32

        # more minor function tests... (overwork today and stop here)
    end
end

end # module
