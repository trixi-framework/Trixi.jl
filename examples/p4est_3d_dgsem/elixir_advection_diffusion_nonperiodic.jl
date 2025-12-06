using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation

diffusivity() = 1 / 17
advection_velocity = (1.0, 0.0, 0.0)
equations = LinearScalarAdvectionEquation3D(advection_velocity)
equations_parabolic = LaplaceDiffusion3D(diffusivity(), equations)

polydeg = 3
solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs)
solver_parabolic = ViscousFormulationLocalDG()

coordinates_min = (-1.0, -0.5, -0.5)
coordinates_max = (0.0, 0.5, 0.5)

# Affine type mapping to take the [-1,1]^3 domain
# and warp it as described in https://arxiv.org/abs/2012.12040
function mapping(xi, eta, zeta)
    y_ = eta + 1 / 6 * (cos(1.5 * pi * xi) * cos(0.5 * pi * eta) * cos(0.5 * pi * zeta))
    x_ = xi + 1 / 6 * (cos(0.5 * pi * xi) * cos(2 * pi * y_) * cos(0.5 * pi * zeta))
    z_ = zeta + 1 / 6 * (cos(0.5 * pi * x_) * cos(pi * y_) * cos(0.5 * pi * zeta))

    # Map from [-1, 1]^3 to [-1, -0.5, -0.5] x [0, 0.5, 0.5]
    x = 0.5 * (x_ - 1)
    y = 0.5 * y_
    z = 0.5 * z_

    return SVector(x, y, z)
end

trees_per_dimension = (5, 3, 3)
mesh = P4estMesh(trees_per_dimension, polydeg = polydeg,
                 mapping = mapping, periodicity = false)

# Example setup taken from
# - Truman Ellis, Jesse Chan, and Leszek Demkowicz (2016).
#   Robust DPG methods for transient convection-diffusion.
#   In: Building bridges: connections and challenges in modern approaches
#   to numerical partial differential equations.
#   [DOI](https://doi.org/10.1007/978-3-319-41640-3_6).
function initial_condition_eriksson_johnson(x, t, equations)
    l = 4
    epsilon = diffusivity() # NOTE: this requires epsilon <= 1/16 due to sqrt
    lambda_1 = (-1 + sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
    lambda_2 = (-1 - sqrt(1 - 4 * epsilon * l)) / (-2 * epsilon)
    r1 = (1 + sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
    s1 = (1 - sqrt(1 + 4 * pi^2 * epsilon^2)) / (2 * epsilon)
    u = exp(-l * t) * (exp(lambda_1 * x[1]) - exp(lambda_2 * x[1])) +
        cos(pi * x[2]) * (exp(s1 * x[1]) - exp(r1 * x[1])) / (exp(-s1) - exp(-r1))
    return SVector(u)
end
initial_condition = initial_condition_eriksson_johnson

boundary_conditions = boundary_condition_default(mesh,
                                                 BoundaryConditionDirichlet(initial_condition))

semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver;
                                             solver_parabolic = solver_parabolic,
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1.0e-11
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
