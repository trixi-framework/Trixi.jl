module TestPerformanceSpecializations2D

using Test
using Trixi
using Trixi: @muladd

include("test_trixi.jl")

EXAMPLES_DIR = examples_dir()

# Start with a clean environment: remove Trixi.jl output directory if it exists
outdir = "out"
isdir(outdir) && rm(outdir, recursive = true)

@testset "Performance specializations 2D" begin
#! format: noindent

@timed_testset "TreeMesh2D, flux_shima_etal_turbo" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "tree_2d_dgsem", "elixir_euler_ec.jl"),
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
        du_specialized = du[:, :, :, 1]

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
        du_baseline = du[:, :, :, 1]

        @test du_specialized ≈ du_baseline
    end
end

@timed_testset "TreeMesh2D, flux_ranocha_turbo" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "tree_2d_dgsem", "elixir_euler_ec.jl"),
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
        du_specialized = du[:, :, :, 1]

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
        du_baseline = du[:, :, :, 1]

        @test du_specialized ≈ du_baseline
    end
end

@timed_testset "StructuredMesh2D, flux_shima_etal_turbo" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "structured_2d_dgsem", "elixir_euler_ec.jl"),
                  cells_per_dimension = (1, 1), tspan = (0.0, 0.0), polydeg = 3,
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
        du_specialized = du[:, :, :, 1]

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
        du_baseline = du[:, :, :, 1]

        @test du_specialized ≈ du_baseline
    end
end

@timed_testset "StructuredMesh2D, flux_ranocha_turbo" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "structured_2d_dgsem", "elixir_euler_ec.jl"),
                  cells_per_dimension = (1, 1), tspan = (0.0, 0.0), polydeg = 3,
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
        du_specialized = du[:, :, :, 1]

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
        du_baseline = du[:, :, :, 1]

        @test du_specialized ≈ du_baseline
    end
end
@timed_testset "P4estMesh2D, combine_conservative_and_nonconservative_fluxes" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "p4est_2d_dgsem", "elixir_mhd_alfven_wave.jl"))
    u_ode = copy(sol.u[end])

    @muladd @inline function flux_central_nonconservative_powell(u_ll, u_rr,
                                                                 normal_direction,
                                                                 equations)
        f = flux_central(u_ll, u_rr, normal_direction, equations)

        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

        inv_rho_ll = 1 / rho_ll
        inv_rho_rr = 1 / rho_rr

        v1_ll = rho_v1_ll * inv_rho_ll
        v2_ll = rho_v2_ll * inv_rho_ll
        v3_ll = rho_v3_ll * inv_rho_ll
        v1_rr = rho_v1_rr * inv_rho_rr
        v2_rr = rho_v2_rr * inv_rho_rr
        v3_rr = rho_v3_rr * inv_rho_rr
        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
        B_dot_n_rr = B1_rr * normal_direction[1] +
                     B2_rr * normal_direction[2]

        B_dot_n_ll = B1_ll * normal_direction[1] +
                     B2_ll * normal_direction[2]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
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

    @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_central_nonconservative_powell),
    equations::IdealGlmMhdEquations2D) = Trixi.True()

    @muladd @inline function flux_hlle_nonconservative_powell(u_ll, u_rr,
                                                              normal_direction,
                                                              equations)
        f = flux_hlle(u_ll, u_rr, normal_direction, equations)

        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

        inv_rho_ll = 1 / rho_ll
        inv_rho_rr = 1 / rho_rr

        v1_ll = rho_v1_ll * inv_rho_ll
        v2_ll = rho_v2_ll * inv_rho_ll
        v3_ll = rho_v3_ll * inv_rho_ll
        v1_rr = rho_v1_rr * inv_rho_rr
        v2_rr = rho_v2_rr * inv_rho_rr
        v3_rr = rho_v3_rr * inv_rho_rr
        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
        B_dot_n_rr = B1_rr * normal_direction[1] +
                     B2_rr * normal_direction[2]

        B_dot_n_ll = B1_ll * normal_direction[1] +
                     B2_ll * normal_direction[2]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
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
    equations::IdealGlmMhdEquations2D) = Trixi.True()

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "p4est_2d_dgsem", "elixir_mhd_alfven_wave.jl"),
                  surface_flux = flux_hlle_nonconservative_powell,
                  volume_flux = flux_central_nonconservative_powell)

    u_ode_specialized = copy(sol.u[end])
    @test u_ode_specialized ≈ u_ode
end
@timed_testset "StructuredMesh2D, combine_conservative_and_nonconservative_fluxes" begin
    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "structured_2d_dgsem",
                           "elixir_mhd_alfven_wave.jl"))
    u_ode = copy(sol.u[end])

    @muladd @inline function flux_lax_friedrichs_nonconservative_powell(u_ll, u_rr,
                                                                        normal_direction,
                                                                        equations)
        f = FluxLaxFriedrichs(max_abs_speed_naive)(u_ll, u_rr, normal_direction,
                                                   equations)

        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

        inv_rho_ll = 1 / rho_ll
        inv_rho_rr = 1 / rho_rr

        v1_ll = rho_v1_ll * inv_rho_ll
        v2_ll = rho_v2_ll * inv_rho_ll
        v3_ll = rho_v3_ll * inv_rho_ll
        v1_rr = rho_v1_rr * inv_rho_rr
        v2_rr = rho_v2_rr * inv_rho_rr
        v3_rr = rho_v3_rr * inv_rho_rr
        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
        B_dot_n_rr = B1_rr * normal_direction[1] +
                     B2_rr * normal_direction[2]

        B_dot_n_ll = B1_ll * normal_direction[1] +
                     B2_ll * normal_direction[2]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
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

    @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_lax_friedrichs_nonconservative_powell),
    equations::IdealGlmMhdEquations2D) = Trixi.True()

    @muladd @inline function flux_central_nonconservative_powell(u_ll, u_rr,
                                                                 normal_direction,
                                                                 equations)
        f = flux_central(u_ll, u_rr, normal_direction, equations)

        rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
        rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

        inv_rho_ll = 1 / rho_ll
        inv_rho_rr = 1 / rho_rr

        v1_ll = rho_v1_ll * inv_rho_ll
        v2_ll = rho_v2_ll * inv_rho_ll
        v3_ll = rho_v3_ll * inv_rho_ll
        v1_rr = rho_v1_rr * inv_rho_rr
        v2_rr = rho_v2_rr * inv_rho_rr
        v3_rr = rho_v3_rr * inv_rho_rr
        v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll
        v_dot_B_rr = v1_rr * B1_rr + v2_rr * B2_rr + v3_rr * B3_rr

        v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
        B_dot_n_rr = B1_rr * normal_direction[1] +
                     B2_rr * normal_direction[2]

        B_dot_n_ll = B1_ll * normal_direction[1] +
                     B2_ll * normal_direction[2]
        v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
        # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
        # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
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

    @inline Trixi.combine_conservative_and_nonconservative_fluxes(::typeof(flux_central_nonconservative_powell),
    equations::IdealGlmMhdEquations2D) = Trixi.True()

    trixi_include(@__MODULE__,
                  joinpath(EXAMPLES_DIR, "structured_2d_dgsem",
                           "elixir_mhd_alfven_wave.jl"),
                  surface_flux = flux_lax_friedrichs_nonconservative_powell,
                  volume_flux = flux_central_nonconservative_powell)

    u_ode_specialized = copy(sol.u[end])
    @test u_ode_specialized ≈ u_ode
end
end

# Clean up afterwards: delete Trixi.jl output directory
@test_nowarn rm(outdir, recursive = true)

end #module
