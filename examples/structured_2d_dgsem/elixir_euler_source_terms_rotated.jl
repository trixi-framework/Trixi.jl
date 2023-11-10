
using OrdinaryDiffEq
using Trixi

# Define new structs inside a module to allow re-evaluating the file.
# This module name needs to be unique among all examples, otherwise Julia will throw warnings
# if multiple test cases using the same module name are run in the same session.
module TrixiExtensionEulerRotated

using Trixi

# initial_condition_convergence_test transformed to the rotated rectangle
struct InitialConditionSourceTermsRotated
    sin_alpha::Float64
    cos_alpha::Float64
end

function InitialConditionSourceTermsRotated(alpha)
    sin_alpha, cos_alpha = sincos(alpha)

    InitialConditionSourceTermsRotated(sin_alpha, cos_alpha)
end

function (initial_condition::InitialConditionSourceTermsRotated)(x, t,
                                                                 equations::CompressibleEulerEquations2D)
    sin_ = initial_condition.sin_alpha
    cos_ = initial_condition.cos_alpha

    # Rotate back to unit square and translate from [-1, 1]^2 to [0, 2]^2

    # Clockwise rotation by α and translation by 1
    # Multiply with [  cos(α)  sin(α);
    #                 -sin(α)  cos(α)]
    x1 = cos_ * x[1] + sin_ * x[2] + 1
    x2 = -sin_ * x[1] + cos_ * x[2] + 1

    rho, rho_v1, rho_v2, rho_e = initial_condition_convergence_test(SVector(x1, x2), t,
                                                                    equations)

    # Rotate velocity vector counterclockwise
    # Multiply with [ cos(α)  -sin(α);
    #                 sin(α)   cos(α)]
    rho_v1_rot = cos_ * rho_v1 - sin_ * rho_v2
    rho_v2_rot = sin_ * rho_v1 + cos_ * rho_v2

    return SVector(rho, rho_v1_rot, rho_v2_rot, rho_e)
end

@inline function (source_terms::InitialConditionSourceTermsRotated)(u, x, t,
                                                                    equations::CompressibleEulerEquations2D)
    sin_ = source_terms.sin_alpha
    cos_ = source_terms.cos_alpha

    # Rotate back to unit square and translate from [-1, 1]^2 to [0, 2]^2

    # Clockwise rotation by α and translation by 1
    # Multiply with [  cos(α)  sin(α);
    #                 -sin(α)  cos(α)]
    x1 = cos_ * x[1] + sin_ * x[2] + 1
    x2 = -sin_ * x[1] + cos_ * x[2] + 1

    du1, du2, du3, du4 = source_terms_convergence_test(u, SVector(x1, x2), t, equations)

    # Rotate velocity vector counterclockwise
    # Multiply with [ cos(α)  -sin(α);
    #                 sin(α)   cos(α)]
    du2_rotated = cos_ * du2 - sin_ * du3
    du3_rotated = sin_ * du2 + cos_ * du3

    return SVector(du1, du2_rotated, du3_rotated, du4)
end

end # module TrixiExtensionEulerRotated

import .TrixiExtensionEulerRotated

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

alpha = 0.1
initial_condition_source_terms = TrixiExtensionEulerRotated.InitialConditionSourceTermsRotated(alpha)
sin_ = initial_condition_source_terms.sin_alpha
cos_ = initial_condition_source_terms.cos_alpha
T = [cos_ -sin_; sin_ cos_]

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

mapping(xi, eta) = T * SVector(xi, eta)

cells_per_dimension = (16, 16)

mesh = StructuredMesh(cells_per_dimension, mapping)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_source_terms, solver,
                                    source_terms = initial_condition_source_terms)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
