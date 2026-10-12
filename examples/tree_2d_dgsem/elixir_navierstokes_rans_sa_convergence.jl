using OrdinaryDiffEqLowStorageRK
using ForwardDiff
using Trixi

###############################################################################
# semidiscretization of the compressible RANS equations with the negative
# Spalart-Allmaras turbulence model (SA-neg)

# Manufactured solution test. The forcing is obtained from the pointwise physical
# fluxes and model source terms by automatic differentiation, so it is consistent
# with the model implementation.

prandtl_number() = 0.72
mu() = 0.01

equations = PassiveTracerEquations(CompressibleEulerEquations2D(1.4), n_tracers = 1)

equations_parabolic = CompressibleRANSDiffusion2D(equations, mu = mu(),
                                                  Prandtl = prandtl_number(),
                                                  model = SpalartAllmarasNeg())

# Smooth, positive wall distance function (no walls are present in this periodic test)
wall_distance(x) = 0.5f0 + 0.2f0 * sinpi(x[1]) * cospi(x[2])

# Manufactured solution in terms of (rho, v1, v2, p, nu_tilde)
function initial_condition_rans_convergence_test(x, t, equations)
    c = 2
    A = 0.1
    ini = c + A * sinpi(x[1] + x[2] - t)

    rho = ini
    v1 = 0.5f0 + 0.1f0 * sinpi(x[1] - 0.5f0 * t) * cospi(x[2])
    v2 = 0.2f0 + 0.1f0 * cospi(x[1]) * sinpi(x[2] + t)
    p = ini^2
    # nu_tilde / nu between approximately 1 and 5, so that all model functions are active
    nu_tilde = mu() / rho * (3 + 2 * sinpi(x[1] - t) * sinpi(x[2]))

    return prim2cons(SVector(rho, v1, v2, p, nu_tilde), equations)
end
initial_condition = initial_condition_rans_convergence_test

# Residual of the full model evaluated for the manufactured solution
source_terms_rans_convergence_test = let equations_parabolic = equations_parabolic
    function (u, x, t, equations)
        u_exact(x_, t_) = initial_condition_rans_convergence_test(x_, t_, equations)
        prim(x_) = Trixi.cons2prim_temperature(u_exact(x_, t), equations_parabolic)
        gradients_prim(x_) = (ForwardDiff.derivative(s -> prim(SVector(s, x_[2])), x_[1]),
                              ForwardDiff.derivative(s -> prim(SVector(x_[1], s)), x_[2]))
        flux_parabolic(x_, orientation) = flux(prim(x_), gradients_prim(x_), orientation,
                                               equations_parabolic)

        du_dt = ForwardDiff.derivative(s -> u_exact(x, s), t)
        div_flux_hyperbolic = ForwardDiff.derivative(s -> flux(u_exact(SVector(s, x[2]),
                                                                       t),
                                                               1, equations), x[1]) +
                              ForwardDiff.derivative(s -> flux(u_exact(SVector(x[1], s),
                                                                       t),
                                                               2, equations), x[2])
        div_flux_parabolic = ForwardDiff.derivative(s -> flux_parabolic(SVector(s, x[2]),
                                                                        1), x[1]) +
                             ForwardDiff.derivative(s -> flux_parabolic(SVector(x[1], s),
                                                                        2), x[2])
        source_model = SourceTermsSpalartAllmaras(wall_distance)(u_exact(x, t),
                                                                 gradients_prim(x), x, t,
                                                                 equations_parabolic)

        return du_dt + div_flux_hyperbolic - div_flux_parabolic - source_model
    end
end

volume_flux = FluxTracerEquationsCentral(flux_ranocha)
polydeg = 3
solver = DGSEM(polydeg = polydeg,
               surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                periodicity = true)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic),
                                             source_terms = source_terms_rans_convergence_test,
                                             source_terms_parabolic = SourceTermsSpalartAllmaras(wall_distance))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-9
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
