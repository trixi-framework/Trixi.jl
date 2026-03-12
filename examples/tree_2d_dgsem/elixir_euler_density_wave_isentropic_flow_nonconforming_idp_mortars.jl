using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

gamma = 3.0
equations = CompressibleEulerEquations2D(gamma)

rho_0(x) = 1 + 0.9999999 * sinpi(x)
rho_0_x(x) = pi * 0.9999999 * cospi(x)
function initial_condition_isentropic_flow(x, t, equations::CompressibleEulerEquations2D)
    theta = deg2rad(45)
    rr = x[1] * cos(theta) + x[2] * sin(theta)

    # Solve for the exact solution implicitly using the method of characteristics
    if t > 0
        # Parameters of Newton method
        tol = 1e-14 # Tolerance for Newton method
        max_iter = 10

        # Find x1
        x1 = rr # Initial guess
        for i in 1:max_iter
            x_old = x1
            x1 = x1 - (rr + sqrt(3) * rho_0(x1) * t - x1) / (sqrt(3) * t * rho_0_x(x1) - 1)
            # Check absolute error
            if abs(x1 - x_old) <= tol
                break
            elseif i == max_iter
                println("ERROR: Newton loop for exact solution didn't converge")
            end
        end

        # Find x2
        x2 = rr # Initial guess
        for i in 1:max_iter
            x_old = x2
            x2 = x2 - (rr - sqrt(3) * rho_0(x2) * t - x2) / (-sqrt(3) * t * rho_0_x(x2) - 1)
            # Check absolute error
            if abs(x2 - x_old) <= tol
                break
            elseif i == max_iter
                println("ERROR: Newton loop for exact solution didn't converge")
            end
        end

        # Exact solution
        rho = 0.5 * (rho_0(x1) + rho_0(x2))
        v1 = sqrt(3) * (rho - rho_0(x1))
    else
        rho = rho_0(rr)
        v1 = 0
    end
    v2 = 0
    p = rho^equations.gamma

    return prim2cons(SVector(rho, v1 * cos(theta), v1 * sin(theta), p), equations)
end
initial_condition = initial_condition_isentropic_flow

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                max_iterations_newton = 50)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
mortar = MortarIDP(equations, basis;
                   positivity_variables_cons = ["rho"],
                   positivity_variables_nonlinear = [pressure],
                   pure_low_order = false)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

# 1d problem in literature uses x in [-1,1]
# That corresponds to our 2D problem with 1d function on the diagonal in
# [-1/sqrt(2), 1/sqrt(2)]^2. However, this setup is not periodic, which is
# why I expanded the domain to [-2/sqrt(2), 2/sqrt(2)]^2.
coordinates_min = (-2/sqrt(2), -2/sqrt(2))
coordinates_max = (2/sqrt(2), 2/sqrt(2))
refinement_patches = ((type = "box", coordinates_min = (0.0, -0.5),
                       coordinates_max = (0.5, 0.5)),)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                refinement_patches = refinement_patches,
                n_cells_max = 30_000,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 5,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode,
                  Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
