
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_inviscid_bow(x, t, equations::CompressibleEulerEquations2D)
    rho = 1.4
    p = 1.0
    v1 = 4.0
    v2 = 0.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_inviscid_bow

boundary_condition = BoundaryConditionCharacteristic(initial_condition)
boundary_conditions = (x_neg = boundary_condition,
                       x_pos = boundary_condition_slip_wall,
                       y_neg = boundary_condition,
                       y_pos = boundary_condition)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 5
basis = LobattoLegendreBasis(polydeg)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                local_twosided_variables_cons = ["rho"],
                                local_onesided_variables_nonlinear = [(Trixi.entropy_guermond_etal,
                                                                       min)],
                                max_iterations_newton = 100,
                                bar_states = true)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# domain
#         ,,
#       ,  |
#     ,    | f4 of length a-1
#f1 ,      |
#  ,     ,`  f2            /alpha
# ,     |_(0,0)___________/_______(3.85,0)
#  ,    |
#   ,   `,
#     ,   `|
#       ,  | f3
#         ,|
# l = circumference of quarter circle / length of egg-shaped form
a = sqrt(5.9^2 - 3.85^2)
alpha = acos(3.85 / 5.9)
l = (pi / 4) / (pi / 2 + 1)
f1(s) = SVector(5.9 * cos(pi - s * alpha) + 3.85, 5.9 * sin(pi - s * alpha)) # left
function f2(s) # right
    t = 0.5 * s + 0.5 # in [0,1]
    if 0 <= t <= l
        beta = t / l # in [0,1]
        return SVector(0.5 * cos(1.5 * pi - beta * 0.5 * pi),
                       0.5 * sin(1.5 * pi - beta * 0.5 * pi) - 0.5)
    elseif l < t <= 1 - l # 0 <= t - l <= 1-2l
        beta = (t - l) / (1 - 2 * l) # in [0,1]
        return SVector(-0.5, -0.5 + beta)
    else # 1 - l < t <= 1
        beta = (t + l - 1) / l # in [0,1]
        return SVector(0.5 * cos(pi - beta * 0.5 * pi),
                       0.5 * sin(pi - beta * 0.5 * pi) + 0.5)
    end
end
f3(s) = SVector(0.0, (a - 1.0) * 0.5 * (s + 1.0) - a) # bottom
f4(s) = SVector(0.0, -(a - 1.0) * 0.5 * (s + 1.0) + a) # top
faces = (f1, f2, f3, f4)

# This creates a mapping that transforms [-1, 1]^2 to the domain with the faces defined above.
Trixi.validate_faces(faces)
mapping_bow = Trixi.transfinite_mapping(faces)

mapping_as_string = "a = sqrt(5.9^2 - 3.85^2); alpha = acos(3.85 / 5.9); l = (pi / 4) / (pi / 2 + 1); " *
                    "f1(s) = SVector(5.9 * cos(pi - s * alpha) + 3.85, 5.9 * sin(pi - s * alpha)); " *
                    "function f2(s); t = 0.5 * s + 0.5; " *
                    "if 0 <= t <= l; beta = t / l; return SVector(0.5 * cos(1.5 * pi - beta * 0.5 * pi), 0.5 * sin(1.5 * pi - beta * 0.5 * pi) - 0.5); " *
                    "elseif l < t <= 1 - l; beta = (t - l) / (1 - 2 * l); return SVector(-0.5, -0.5 + beta); " *
                    "else beta = (t + l - 1) / l; return SVector(0.5 * cos(pi - beta * 0.5 * pi), 0.5 * sin(pi - beta * 0.5 * pi) + 0.5); end; end; " *
                    "f3(s) = SVector(0.0,  (a - 1.0) * 0.5 * (s + 1.0) - a); " *
                    "f4(s) = SVector(0.0, -(a - 1.0) * 0.5 * (s + 1.0) + a); " *
                    "faces = (f1, f2, f3, f4); mapping = Trixi.transfinite_mapping(faces)"

cells_per_dimension = (8, 12)

mesh = StructuredMesh(cells_per_dimension, mapping_bow,
                      mapping_as_string = mapping_as_string, periodicity = false)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 2000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        save_solution)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback(save_errors = false))

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
summary_callback() # print the timer summary
