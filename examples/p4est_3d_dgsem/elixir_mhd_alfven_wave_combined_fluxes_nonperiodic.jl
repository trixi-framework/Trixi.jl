using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: @muladd
###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
#
# In this elixir, we employ the specialization that combines conservative and
# nonconservative fluxes in a single kernel, avoiding repeated operations such as
# computing the primitive variables and multiple mean values for both fluxes.
# The single kernel must return the two contributions
#
#   flux_cons(u_ll, u_rr, normal_direction, equations)
#     + 0.5f0 * flux_noncons(u_ll, u_rr, normal_direction, equations)
#
# and
#
#   flux_cons(u_ll, u_rr, normal_direction, equations)
#     + 0.5f0 * flux_noncons(u_rr, u_ll, normal_direction, equations),
#
# as shown below.
@muladd @inline function flux_hindenlang_gassner_nonconservative_powell(u_ll, u_rr,
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
    rho_mean = Trixi.ln_mean(rho_ll, rho_rr)
    # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
    # in exact arithmetic since
    #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
    #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
    inv_rho_p_mean = p_ll * p_rr * Trixi.inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
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

@muladd @inline function flux_hlle_nonconservative_powell(u_ll, u_rr,
                                                          normal_direction::AbstractVector,
                                                          equations::IdealGlmMhdEquations3D)
    f = flux_hlle(u_ll, u_rr, normal_direction, equations)

    rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_total_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_total_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

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

# For nonperiodic boundary conditions, the boundary flux must also be specialized
# such that it returns only the first contribution of the surface flux function.
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

equations = IdealGlmMhdEquations3D(5 / 3)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; x_neg = boundary_condition,
                       x_pos = boundary_condition,
                       y_neg = boundary_condition,
                       y_pos = boundary_condition,
                       z_neg = boundary_condition,
                       z_pos = boundary_condition)

solver = DGSEM(polydeg = 3,
               surface_flux = flux_hlle_nonconservative_powell,
               volume_integral = VolumeIntegralFluxDifferencing(flux_hindenlang_gassner_nonconservative_powell))

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

# Create P4estMesh with 2 x 2 x 2 trees
trees_per_dimension = (2, 2, 2)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3, initial_refinement_level = 2,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 1.0
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
