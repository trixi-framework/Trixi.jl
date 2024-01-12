
using OrdinaryDiffEq
using Trixi
using LinearAlgebra: norm, dot # for use in the MHD boundary condition

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
equations = IdealGlmMhdEquations2D(1.4)

function initial_condition_perturbation(x, t, equations::IdealGlmMhdEquations2D)
    # pressure perturbation in a vertically magnetized field on the domain [-1, 1]^2

    r2 = (x[1] + 0.25)^2 + (x[2] + 0.25)^2

    rho = 1.0
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    p = 1 + 0.5 * exp(-100 * r2)

    # the pressure and magnetic field are chosen to be strongly
    # magnetized, such that p / ||B||^2 ≈ 0.01.
    B1 = 0.0
    B2 = 40.0 / sqrt(4.0 * pi)
    B3 = 0.0

    psi = 0.0
    return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end
initial_condition = initial_condition_perturbation

surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)

solver = DGMulti(polydeg = 3, element_type = Quad(), approximation_type = GaussSBP(),
                 surface_integral = SurfaceIntegralWeakForm(surface_flux),
                 volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

x_neg(x, tol = 50 * eps()) = abs(x[1] + 1) < tol
x_pos(x, tol = 50 * eps()) = abs(x[1] - 1) < tol
y_neg(x, tol = 50 * eps()) = abs(x[2] + 1) < tol
y_pos(x, tol = 50 * eps()) = abs(x[2] - 1) < tol
is_on_boundary = Dict(:x_neg => x_neg, :x_pos => x_pos, :y_neg => y_neg, :y_pos => y_pos)

cells_per_dimension = (16, 16)
mesh = DGMultiMesh(solver, cells_per_dimension; periodicity = (false, false),
                   is_on_boundary)

# Create a "reflective-like" boundary condition by mirroring the velocity but leaving the magnetic field alone.
# Note that this boundary condition is probably not entropy stable.
function boundary_condition_velocity_slip_wall(u_inner, normal_direction::AbstractVector,
                                               x, t, surface_flux_function,
                                               equations::IdealGlmMhdEquations2D)

    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    norm_ = norm(normal_direction)
    normal = normal_direction / norm_

    # compute the primitive variables
    rho, v1, v2, v3, p, B1, B2, B3, psi = cons2prim(u_inner, equations)

    v_normal = dot(normal, SVector(v1, v2))
    u_mirror = prim2cons(SVector(rho, v1 - 2 * v_normal * normal[1],
                                 v2 - 2 * v_normal * normal[2],
                                 v3, p, B1, B2, B3, psi), equations)

    return surface_flux_function(u_inner, u_mirror, normal, equations) * norm_
end

boundary_conditions = (; x_neg = boundary_condition_velocity_slip_wall,
                       x_pos = boundary_condition_velocity_slip_wall,
                       y_neg = boundary_condition_do_nothing,
                       y_pos = BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.075)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     uEltype = real(solver))
alive_callback = AliveCallback(alive_interval = 10)

cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)
glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1e-5, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
