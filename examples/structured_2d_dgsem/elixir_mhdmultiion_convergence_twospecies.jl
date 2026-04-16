using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
"""
  electron_pressure_alpha(u, equations::IdealGlmMhdMultiIonEquations2D)
Returns a fraction (alpha) of the total ion pressure for the electron pressure.
"""
function electron_pressure_alpha(u, equations::IdealGlmMhdMultiIonEquations2D)
    alpha = 0.2
    prim = cons2prim(u, equations)
    p_e = zero(u[1])
    for k in eachcomponent(equations)
        _, _, _, _, p_k = Trixi.get_component(k, prim, equations)
        p_e += p_k
    end
    return alpha * p_e
end
# semidiscretization of the ideal multi-ion MHD equations
equations = IdealGlmMhdMultiIonEquations2D(gammas = (2.0, 4.0),
                                           charge_to_mass = (2.0, 1.0),
                                           electron_pressure = electron_pressure_alpha)

"""
Initial (and exact) solution for the the manufactured solution test. Runs with 
* gammas = (2.0, 4.0),
* charge_to_mass = (2.0, 1.0)
* Domain size: [-1,1]²
"""
function initial_condition_manufactured_solution(x, t,
                                                 equations::IdealGlmMhdMultiIonEquations2D)
    am = 0.1
    om = π
    h = am * sin(om * (x[1] + x[2] - t)) + 2
    hh1 = am * 0.4 * sin(om * (x[1] + x[2] - t)) + 1
    hh2 = h - hh1

    u1 = hh1
    u2 = hh1
    u3 = hh1
    u4 = 0.1 * hh1
    u5 = 2 * hh1^2 + hh1
    u6 = hh2
    u7 = hh2
    u8 = hh2
    u9 = 0.1 * hh2
    u10 = 2 * hh2^2 + hh2
    u11 = 0.25 * h
    u12 = -0.25 * h
    u13 = 0.1 * h

    return SVector{nvariables(equations), real(equations)}(u11, u12, u13,
                                                           u1, u2, u3, u4, u5,
                                                           u6, u7, u8, u9, u10,
                                                           0)
end

"""
Source term that corresponds to the manufactured solution test. Runs with 
* gammas = (2.0, 4.0),
* charge_to_mass = (2.0, 1.0)
* Domain size: [-1,1]²
"""
function source_terms_manufactured_solution_pe(u, x, t,
                                               equations::IdealGlmMhdMultiIonEquations2D)
    am = 0.1
    om = pi
    h1 = am * sin(om * (x[1] + x[2] - t))
    hx = am * om * cos(om * (x[1] + x[2] - t))

    s1 = (2 * hx) / 5
    s2 = (38055 * hx * h1^2 + 185541 * hx * h1 + 220190 * hx) / (35000 * h1 + 75000)
    s3 = (38055 * hx * h1^2 + 185541 * hx * h1 + 220190 * hx) / (35000 * h1 + 75000)
    s4 = hx / 25
    s5 = (1835811702576186755 * hx * h1^2 + 8592627463681183181 * hx * h1 +
          9884050459977240490 * hx) / (652252660543767500 * h1 + 1397684272593787500)
    s6 = (3 * hx) / 5
    s7 = (76155 * hx * h1^2 + 295306 * hx * h1 + 284435 * hx) / (17500 * h1 + 37500)
    s8 = (76155 * hx * h1^2 + 295306 * hx * h1 + 284435 * hx) / (17500 * h1 + 37500)
    s9 = (3 * hx) / 50
    s10 = (88755 * hx * h1^2 + 338056 * hx * h1 + 318185 * hx) / (8750 * h1 + 18750)
    s11 = hx / 4
    s12 = -hx / 4
    s13 = hx / 10

    s = SVector{nvariables(equations), real(equations)}(s11, s12, s13,
                                                        s1, s2, s3, s4, s5,
                                                        s6, s7, s8, s9, s10,
                                                        0)
    S_std = source_terms_lorentz(u, x, t, equations::IdealGlmMhdMultiIonEquations2D)

    return SVector{nvariables(equations), real(equations)}(S_std .+ s)
end

initial_condition = initial_condition_manufactured_solution
source_terms = source_terms_manufactured_solution_pe

volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive),
                flux_nonconservative_central) # Also works with flux_nonconservative_ruedaramirez_etal

solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

# To test convergence use:
#   convergence_test("../examples/structured_2d_dgsem/elixir_mhdmultiion_convergence_twospecies.jl", 3, cells_per_dimension = (2, 2), polydeg = 3)
# Mapping as described in https://arxiv.org/abs/2012.12040
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5

    y = eta +
        0.05 * (cospi(1.5 * (2 * xi - 3) / 3) *
                cospi(0.5 * (2 * eta - 3) / 3))

    x = xi +
        0.05 * (cospi(0.5 * (2 * xi - 3) / 3) *
                cospi(2 * (2 * y - 3) / 3))

    # Go back to [-1,1]^3
    x = x * 2 / 3 - 1
    y = y * 2 / 3 - 1

    return SVector(x, y)
end
cells_per_dimension = (2, 2)
mesh = StructuredMesh(cells_per_dimension, mapping;
                      periodicity = true)

# # Alternatively, you can test with a TreeMesh
# #   convergence_test("../examples/structured_2d_dgsem/elixir_mhdmultiion_convergence_twospecies.jl", 3, initial_refinement_level = 1, polydeg = 3)
# initial_refinement_level = 1
# mesh = TreeMesh(coordinates_min, coordinates_max,
#                 initial_refinement_level = initial_refinement_level,
#                 n_cells_max = 1_000_000,
#                 periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
