module TestPerformanceSpecializations3D

using Test
using Trixi

include("test_trixi.jl")

EXAMPLES_DIR = examples_dir()

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "Performance specializations 3D" begin
#! format: noindent

@timed_testset "TreeMesh3D, flux_shima_etal_turbo" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "tree_3d_dgsem", "elixir_euler_ec.jl"),
                  initial_refinement_level = 0, tspan = (0.0, 0.0), polydeg = 3,
                  volume_flux = flux_shima_etal_turbo,
                  surface_flux = flux_shima_etal_turbo)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)

    # Preserve original memory since it will be `unsafe_wrap`ped and might
    # thus otherwise be garbage collected
    GC.@preserve u_ode du_ode begin
        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)
        have_nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

        # Call the optimized default version
        du .= 0
        Trixi.flux_differencing_kernel!(du, u, 1, semi.mesh,
                                        have_nonconservative_terms, semi.equations,
                                        semi.solver.volume_integral.volume_flux,
                                        semi.solver, semi.cache, true)
        du_specialized = du[:, :, :, :, 1]

        # Call the plain version - note the argument type `Function` of
        # `semi.solver.volume_integral.volume_flux`
        du .= 0
        invoke(Trixi.flux_differencing_kernel!,
               Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
                     typeof(have_nonconservative_terms), typeof(semi.equations),
                     Function, typeof(semi.solver), typeof(semi.cache), Bool},
               du, u, 1, semi.mesh,
               have_nonconservative_terms, semi.equations,
               semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
        du_baseline = du[:, :, :, :, 1]

        @test du_specialized ≈ du_baseline
    end
end

@timed_testset "TreeMesh3D, flux_ranocha_turbo" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "tree_3d_dgsem", "elixir_euler_ec.jl"),
                  initial_refinement_level = 0, tspan = (0.0, 0.0), polydeg = 3,
                  volume_flux = flux_ranocha_turbo, surface_flux = flux_ranocha_turbo)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)

    # Preserve original memory since it will be `unsafe_wrap`ped and might
    # thus otherwise be garbage collected
    GC.@preserve u_ode du_ode begin
        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)
        have_nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

        # Call the optimized default version
        du .= 0
        Trixi.flux_differencing_kernel!(du, u, 1, semi.mesh,
                                        have_nonconservative_terms, semi.equations,
                                        semi.solver.volume_integral.volume_flux,
                                        semi.solver, semi.cache, true)
        du_specialized = du[:, :, :, :, 1]

        # Call the plain version - note the argument type `Function` of
        # `semi.solver.volume_integral.volume_flux`
        du .= 0
        invoke(Trixi.flux_differencing_kernel!,
               Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
                     typeof(have_nonconservative_terms), typeof(semi.equations),
                     Function, typeof(semi.solver), typeof(semi.cache), Bool},
               du, u, 1, semi.mesh,
               have_nonconservative_terms, semi.equations,
               semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
        du_baseline = du[:, :, :, :, 1]

        @test du_specialized ≈ du_baseline
    end
end

@timed_testset "StructuredMesh3D, flux_shima_etal_turbo" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "structured_3d_dgsem", "elixir_euler_ec.jl"),
                  cells_per_dimension = (1, 1, 1), tspan = (0.0, 0.0), polydeg = 3,
                  volume_flux = flux_shima_etal_turbo,
                  surface_flux = flux_shima_etal_turbo)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)

    # Preserve original memory since it will be `unsafe_wrap`ped and might
    # thus otherwise be garbage collected
    GC.@preserve u_ode du_ode begin
        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)
        have_nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

        # Call the optimized default version
        du .= 0
        Trixi.flux_differencing_kernel!(du, u, 1, semi.mesh,
                                        have_nonconservative_terms, semi.equations,
                                        semi.solver.volume_integral.volume_flux,
                                        semi.solver, semi.cache, true)
        du_specialized = du[:, :, :, :, 1]

        # Call the plain version - note the argument type `Function` of
        # `semi.solver.volume_integral.volume_flux`
        du .= 0
        invoke(Trixi.flux_differencing_kernel!,
               Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
                     typeof(have_nonconservative_terms), typeof(semi.equations),
                     Function, typeof(semi.solver), typeof(semi.cache), Bool},
               du, u, 1, semi.mesh,
               have_nonconservative_terms, semi.equations,
               semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
        du_baseline = du[:, :, :, :, 1]

        @test du_specialized ≈ du_baseline
    end
end

@timed_testset "StructuredMesh3D, flux_ranocha_turbo" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "structured_3d_dgsem", "elixir_euler_ec.jl"),
                  cells_per_dimension = (1, 1, 1), tspan = (0.0, 0.0), polydeg = 3,
                  volume_flux = flux_ranocha_turbo, surface_flux = flux_ranocha_turbo)
    u_ode = copy(sol.u[end])
    du_ode = zero(u_ode)

    # Preserve original memory since it will be `unsafe_wrap`ped and might
    # thus otherwise be garbage collected
    GC.@preserve u_ode du_ode begin
        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)
        have_nonconservative_terms = Trixi.have_nonconservative_terms(semi.equations)

        # Call the optimized default version
        du .= 0
        Trixi.flux_differencing_kernel!(du, u, 1, semi.mesh,
                                        have_nonconservative_terms, semi.equations,
                                        semi.solver.volume_integral.volume_flux,
                                        semi.solver, semi.cache, true)
        du_specialized = du[:, :, :, :, 1]

        # Call the plain version - note the argument type `Function` of
        # `semi.solver.volume_integral.volume_flux`
        du .= 0
        invoke(Trixi.flux_differencing_kernel!,
               Tuple{typeof(du), typeof(u), Integer, typeof(semi.mesh),
                     typeof(have_nonconservative_terms), typeof(semi.equations),
                     Function, typeof(semi.solver), typeof(semi.cache), Bool},
               du, u, 1, semi.mesh,
               have_nonconservative_terms, semi.equations,
               semi.solver.volume_integral.volume_flux, semi.solver, semi.cache, true)
        du_baseline = du[:, :, :, :, 1]

        @test du_specialized ≈ du_baseline
    end
end

@timed_testset "P4estMesh3D, combine_conservative_and_nonconservative_fluxes" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                           "elixir_mhd_alfven_wave_nonperiodic.jl"),
                  volume_integral = VolumeIntegralFluxDifferencing((flux_hindenlang_gassner,
                                                                    flux_nonconservative_powell)),
                  tspan = (0.0, 0.1))
    u_ode = copy(sol.u[end])

    @inline function flux_hindenlang_gassner_nonconservative_powell(u_ll, u_rr,
                                                                    normal_direction::AbstractVector,
                                                                    equations::IdealGlmMhdEquations3D)
        # Unpack left and right states
        rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll,
                                                                                   equations)
        rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr,
                                                                                   equations)
        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] +
                     v3_ll * normal_direction[3]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] +
                     v3_rr * normal_direction[3]
        B_dot_n_ll = B1_ll * normal_direction[1] + B2_ll * normal_direction[2] +
                     B3_ll * normal_direction[3]
        B_dot_n_rr = B1_rr * normal_direction[1] + B2_rr * normal_direction[2] +
                     B3_rr * normal_direction[3]

        # Compute the necessary mean values needed for either direction
        rho_mean = ln_mean(rho_ll, rho_rr)
        # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
        # in exact arithmetic since
        #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
        #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
        inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
        v1_avg = 0.5f0 * (v1_ll + v1_rr)
        v2_avg = 0.5f0 * (v2_ll + v2_rr)
        v3_avg = 0.5f0 * (v3_ll + v3_rr)
        p_avg = 0.5f0 * (p_ll + p_rr)
        psi_avg = 0.5f0 * (psi_ll + psi_rr)
        velocity_square_avg = 0.5f0 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
        magnetic_square_avg = 0.5f0 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

        # Calculate fluxes depending on normal_direction
        f1 = rho_mean * 0.5f0 * (v_dot_n_ll + v_dot_n_rr)
        f2 = (f1 * v1_avg + (p_avg + magnetic_square_avg) * normal_direction[1]
              -
              0.5f0 * (B_dot_n_ll * B1_rr + B_dot_n_rr * B1_ll))
        f3 = (f1 * v2_avg + (p_avg + magnetic_square_avg) * normal_direction[2]
              -
              0.5f0 * (B_dot_n_ll * B2_rr + B_dot_n_rr * B2_ll))
        f4 = (f1 * v3_avg + (p_avg + magnetic_square_avg) * normal_direction[3]
              -
              0.5f0 * (B_dot_n_ll * B3_rr + B_dot_n_rr * B3_ll))
        #f5 below
        f6 = (equations.c_h * psi_avg * normal_direction[1]
              +
              0.5f0 * (v_dot_n_ll * B1_ll - v1_ll * B_dot_n_ll +
               v_dot_n_rr * B1_rr - v1_rr * B_dot_n_rr))
        f7 = (equations.c_h * psi_avg * normal_direction[2]
              +
              0.5f0 * (v_dot_n_ll * B2_ll - v2_ll * B_dot_n_ll +
               v_dot_n_rr * B2_rr - v2_rr * B_dot_n_rr))
        f8 = (equations.c_h * psi_avg * normal_direction[3]
              +
              0.5f0 * (v_dot_n_ll * B3_ll - v3_ll * B_dot_n_ll +
               v_dot_n_rr * B3_rr - v3_rr * B_dot_n_rr))
        f9 = equations.c_h * 0.5f0 * (B_dot_n_ll + B_dot_n_rr)
        # total energy flux is complicated and involves the previous components
        f5 = (f1 *
              (velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one)
              +
              0.5f0 * (+p_ll * v_dot_n_rr + p_rr * v_dot_n_ll
               + (v_dot_n_ll * B1_ll * B1_rr + v_dot_n_rr * B1_rr * B1_ll)
               + (v_dot_n_ll * B2_ll * B2_rr + v_dot_n_rr * B2_rr * B2_ll)
               + (v_dot_n_ll * B3_ll * B3_rr + v_dot_n_rr * B3_rr * B3_ll)
               -
               (v1_ll * B_dot_n_ll * B1_rr + v1_rr * B_dot_n_rr * B1_ll)
               -
               (v2_ll * B_dot_n_ll * B2_rr + v2_rr * B_dot_n_rr * B2_ll)
               -
               (v3_ll * B_dot_n_ll * B3_rr + v3_rr * B_dot_n_rr * B3_ll)
               +
               equations.c_h * (B_dot_n_ll * psi_rr + B_dot_n_rr * psi_ll)))

        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr
        f = SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2,3}, 0, 0, 0, v_{1,2,3})
        g_left = SVector(0,
                         B1_ll * B_dot_n_rr,
                         B2_ll * B_dot_n_rr,
                         B3_ll * B_dot_n_rr,
                         v_dot_B_ll * B_dot_n_rr + v_dot_n_ll * psi_ll * psi_rr,
                         v1_ll * B_dot_n_rr,
                         v2_ll * B_dot_n_rr,
                         v3_ll * B_dot_n_rr,
                         v_dot_n_ll * psi_rr)

        g_right = SVector(0,
                          B1_rr * B_dot_n_ll,
                          B2_rr * B_dot_n_ll,
                          B3_rr * B_dot_n_ll,
                          v_dot_B_rr * B_dot_n_ll + v_dot_n_rr * psi_rr * psi_ll,
                          v1_rr * B_dot_n_ll,
                          v2_rr * B_dot_n_ll,
                          v3_rr * B_dot_n_ll,
                          v_dot_n_rr * psi_ll)
        flux_left = f + 0.5f0 * g_left
        flux_right = f + 0.5f0 * g_right
        return flux_left, flux_right
    end

    @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_hindenlang_gassner_nonconservative_powell),
    equations::IdealGlmMhdEquations3D) = Trixi.True()

    @inline function flux_hlle_nonconservative_powell(u_ll, u_rr,
                                                      normal_direction::AbstractVector,
                                                      equations::IdealGlmMhdEquations3D)
        f = flux_hlle(u_ll, u_rr, normal_direction, equations)

        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

        v1_ll = rho_v1_ll / rho_ll
        v2_ll = rho_v2_ll / rho_ll
        v3_ll = rho_v3_ll / rho_ll
        v1_rr = rho_v1_rr / rho_rr
        v2_rr = rho_v2_rr / rho_rr
        v3_rr = rho_v3_rr / rho_rr
        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] +
                     v3_ll * normal_direction[3]
        B_dot_n_rr = B1_rr * normal_direction[1] +
                     B2_rr * normal_direction[2] +
                     B3_rr * normal_direction[3]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] +
                     v3_rr * normal_direction[3]
        B_dot_n_ll = B1_ll * normal_direction[1] +
                     B2_ll * normal_direction[2] +
                     B3_ll * normal_direction[3]

        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2,3}, 0, 0, 0, v_{1,2,3})
        g_left = SVector(0,
                         B1_ll * B_dot_n_rr,
                         B2_ll * B_dot_n_rr,
                         B3_ll * B_dot_n_rr,
                         v_dot_B_ll * B_dot_n_rr + v_dot_n_ll * psi_ll * psi_rr,
                         v1_ll * B_dot_n_rr,
                         v2_ll * B_dot_n_rr,
                         v3_ll * B_dot_n_rr,
                         v_dot_n_ll * psi_rr)

        g_right = SVector(0,
                          B1_rr * B_dot_n_ll,
                          B2_rr * B_dot_n_ll,
                          B3_rr * B_dot_n_ll,
                          v_dot_B_rr * B_dot_n_ll + v_dot_n_rr * psi_rr * psi_ll,
                          v1_rr * B_dot_n_ll,
                          v2_rr * B_dot_n_ll,
                          v3_rr * B_dot_n_ll,
                          v_dot_n_rr * psi_ll)
        flux_left = f + 0.5f0 * g_left
        flux_right = f + 0.5f0 * g_right

        return flux_left, flux_right
    end

    @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_hlle_nonconservative_powell),
    equations::IdealGlmMhdEquations3D) = Trixi.True()

    @inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner,
                                                                      normal_direction::AbstractVector,
                                                                      x, t,
                                                                      surface_flux_function::typeof(flux_hlle_nonconservative_powell),
                                                                      equations)

        # get the external value of the solution
        u_boundary = boundary_condition.boundary_value_function(x, t, equations)

        # Calculate boundary flux
        flux, _ = surface_flux_function(u_inner, u_boundary, normal_direction,
                                        equations)
        return flux
    end

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "p4est_3d_dgsem",
                           "elixir_mhd_alfven_nonperiodic.jl"),
                  surface_flux = flux_lax_friedrichs_nonconservative_powell,
                  volume_integral = VolumeIntegralFluxDifferencing(flux_hindenlang_gassner_nonconservative_powell),
                  tspan = (0.0, 0.1))

    u_ode_specialized = copy(sol.u[end])
    @test u_ode_specialized ≈ u_ode
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end #module
