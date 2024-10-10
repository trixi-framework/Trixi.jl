
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach038_flow(x, t,
                                                equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 0.38
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach038_flow

volume_flux = flux_ranocha_turbo # FluxRotated(flux_chandrashekar) can also be used
surface_flux = flux_lax_friedrichs

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

function mapping2cylinder(xi, eta)
    xi_, eta_ = 0.5 * (xi + 1), 0.5 * (eta + 1.0) # Map from [-1,1] to [0,1] for simplicity

    R2 = 50.0 # Bigger circle
    R1 = 0.5  # Smaller circle

    # Ensure an isotropic mesh by using elements with smaller radial length near the inner circle

    r = R1 * exp(xi_ * log(R2 / R1))
    theta = 2.0 * pi * eta_

    x = r * cos(theta)
    y = r * sin(theta)
    return (x, y)
end

cells_per_dimension = (64, 64)
# xi = -1 maps to the inner circle and xi = +1 maps to the outer circle and we can specify boundary
# conditions there. However, the image of eta = -1, +1 coincides at the line y = 0. There is no
# physical boundary there so we specify `periodicity = true` there and the solver treats the
# element across eta = -1, +1 as neighbours which is what we want
mesh = P4estMesh(cells_per_dimension, mapping = mapping2cylinder, polydeg = polydeg,
                 periodicity = (false, true))

# The boundary condition on the outer cylinder is constant but subsonic, so we cannot compute the
# boundary flux from the external information alone. Thus, we use the numerical flux to distinguish
# between inflow and outflow characteristics
@inline function boundary_condition_subsonic_constant(u_inner,
                                                      normal_direction::AbstractVector, x,
                                                      t,
                                                      surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach038_flow(x, t, equations)

    return surface_flux_function(u_inner, u_boundary, normal_direction, equations)
end

boundary_conditions = Dict(:x_neg => boundary_condition_slip_wall,
                           :x_pos => boundary_condition_subsonic_constant)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers

# Run for a long time to reach a steady state
tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

# Callbacks

summary_callback = SummaryCallback()

analysis_interval = 2000

aoa = 0.0
rho_inf = 1.4
u_inf = 0.38
l_inf = 1.0 # Diameter of circle

drag_coefficient = AnalysisSurfaceIntegral((:x_neg,),
                                           DragCoefficientPressure(aoa, rho_inf, u_inf,
                                                                   l_inf))

lift_coefficient = AnalysisSurfaceIntegral((:x_neg,),
                                           LiftCoefficientPressure(aoa, rho_inf, u_inf,
                                                                   l_inf))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_integrals = (drag_coefficient,
                                                           lift_coefficient))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 2.0)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation
sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false;
                                 thread = OrdinaryDiffEq.True()),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
