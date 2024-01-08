using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.72
mu() = 0.01

equations = CompressibleEulerEquations3D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())

coordinates_min = (-1.0, -1.0, -1.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (1.0, 1.0, 1.0) # maximum coordinates (max(x), max(y), max(z))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = (true, false, true),
                n_cells_max = 50_000) # set maximum capacity of tree data structure

# Note: the initial condition cannot be specialized to `CompressibleNavierStokesDiffusion3D`
#       since it is called by both the parabolic solver (which passes in `CompressibleNavierStokesDiffusion3D`)
#       and by the initial condition (which passes in `CompressibleEulerEquations3D`).
# This convergence test setup was originally derived by Andrew Winters (@andrewwinters5000)
function initial_condition_navier_stokes_convergence_test(x, t, equations)
    # Constants. OBS! Must match those in `source_terms_navier_stokes_convergence_test`
    c = 2.0
    A1 = 0.5
    A2 = 1.0
    A3 = 0.5

    # Convenience values for trig. functions
    pi_x = pi * x[1]
    pi_y = pi * x[2]
    pi_z = pi * x[3]
    pi_t = pi * t

    rho = c + A1 * sin(pi_x) * cos(pi_y) * sin(pi_z) * cos(pi_t)
    v1 = A2 * sin(pi_x) * log(x[2] + 2.0) * (1.0 - exp(-A3 * (x[2] - 1.0))) * sin(pi_z) *
         cos(pi_t)
    v2 = v1
    v3 = v1
    p = rho^2

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end

@inline function source_terms_navier_stokes_convergence_test(u, x, t, equations)
    # TODO: parabolic
    # we currently need to hardcode these parameters until we fix the "combined equation" issue
    # see also https://github.com/trixi-framework/Trixi.jl/pull/1160
    inv_gamma_minus_one = inv(equations.gamma - 1)
    Pr = prandtl_number()
    mu_ = mu()

    # Constants. OBS! Must match those in `initial_condition_navier_stokes_convergence_test`
    c = 2.0
    A1 = 0.5
    A2 = 1.0
    A3 = 0.5

    # Convenience values for trig. functions
    pi_x = pi * x[1]
    pi_y = pi * x[2]
    pi_z = pi * x[3]
    pi_t = pi * t

    # Define auxiliary functions for the strange function of the y variable
    # to make expressions easier to read
    g = log(x[2] + 2.0) * (1.0 - exp(-A3 * (x[2] - 1.0)))
    g_y = (A3 * log(x[2] + 2.0) * exp(-A3 * (x[2] - 1.0)) +
           (1.0 - exp(-A3 * (x[2] - 1.0))) / (x[2] + 2.0))
    g_yy = (2.0 * A3 * exp(-A3 * (x[2] - 1.0)) / (x[2] + 2.0) -
            (1.0 - exp(-A3 * (x[2] - 1.0))) / ((x[2] + 2.0)^2) -
            A3^2 * log(x[2] + 2.0) * exp(-A3 * (x[2] - 1.0)))

    # Density and its derivatives
    rho = c + A1 * sin(pi_x) * cos(pi_y) * sin(pi_z) * cos(pi_t)
    rho_t = -pi * A1 * sin(pi_x) * cos(pi_y) * sin(pi_z) * sin(pi_t)
    rho_x = pi * A1 * cos(pi_x) * cos(pi_y) * sin(pi_z) * cos(pi_t)
    rho_y = -pi * A1 * sin(pi_x) * sin(pi_y) * sin(pi_z) * cos(pi_t)
    rho_z = pi * A1 * sin(pi_x) * cos(pi_y) * cos(pi_z) * cos(pi_t)
    rho_xx = -pi^2 * (rho - c)
    rho_yy = -pi^2 * (rho - c)
    rho_zz = -pi^2 * (rho - c)

    # Velocities and their derivatives
    # v1 terms
    v1 = A2 * sin(pi_x) * g * sin(pi_z) * cos(pi_t)
    v1_t = -pi * A2 * sin(pi_x) * g * sin(pi_z) * sin(pi_t)
    v1_x = pi * A2 * cos(pi_x) * g * sin(pi_z) * cos(pi_t)
    v1_y = A2 * sin(pi_x) * g_y * sin(pi_z) * cos(pi_t)
    v1_z = pi * A2 * sin(pi_x) * g * cos(pi_z) * cos(pi_t)
    v1_xx = -pi^2 * v1
    v1_yy = A2 * sin(pi_x) * g_yy * sin(pi_z) * cos(pi_t)
    v1_zz = -pi^2 * v1
    v1_xy = pi * A2 * cos(pi_x) * g_y * sin(pi_z) * cos(pi_t)
    v1_xz = pi^2 * A2 * cos(pi_x) * g * cos(pi_z) * cos(pi_t)
    v1_yz = pi * A2 * sin(pi_x) * g_y * cos(pi_z) * cos(pi_t)
    # v2 terms (simplifies from ansatz)
    v2 = v1
    v2_t = v1_t
    v2_x = v1_x
    v2_y = v1_y
    v2_z = v1_z
    v2_xx = v1_xx
    v2_yy = v1_yy
    v2_zz = v1_zz
    v2_xy = v1_xy
    v2_yz = v1_yz
    # v3 terms (simplifies from ansatz)
    v3 = v1
    v3_t = v1_t
    v3_x = v1_x
    v3_y = v1_y
    v3_z = v1_z
    v3_xx = v1_xx
    v3_yy = v1_yy
    v3_zz = v1_zz
    v3_xz = v1_xz
    v3_yz = v1_yz

    # Pressure and its derivatives
    p = rho^2
    p_t = 2.0 * rho * rho_t
    p_x = 2.0 * rho * rho_x
    p_y = 2.0 * rho * rho_y
    p_z = 2.0 * rho * rho_z

    # Total energy and its derivatives; simiplifies from ansatz that v2 = v1 and v3 = v1
    E = p * inv_gamma_minus_one + 1.5 * rho * v1^2
    E_t = p_t * inv_gamma_minus_one + 1.5 * rho_t * v1^2 + 3.0 * rho * v1 * v1_t
    E_x = p_x * inv_gamma_minus_one + 1.5 * rho_x * v1^2 + 3.0 * rho * v1 * v1_x
    E_y = p_y * inv_gamma_minus_one + 1.5 * rho_y * v1^2 + 3.0 * rho * v1 * v1_y
    E_z = p_z * inv_gamma_minus_one + 1.5 * rho_z * v1^2 + 3.0 * rho * v1 * v1_z

    # Divergence of Fick's law ∇⋅∇q = kappa ∇⋅∇T; simplifies because p = rho², so T = p/rho = rho
    kappa = equations.gamma * inv_gamma_minus_one / Pr
    q_xx = kappa * rho_xx # kappa T_xx
    q_yy = kappa * rho_yy # kappa T_yy
    q_zz = kappa * rho_zz # kappa T_zz

    # Stress tensor and its derivatives (exploit symmetry)
    tau11 = 4.0 / 3.0 * v1_x - 2.0 / 3.0 * (v2_y + v3_z)
    tau12 = v1_y + v2_x
    tau13 = v1_z + v3_x
    tau22 = 4.0 / 3.0 * v2_y - 2.0 / 3.0 * (v1_x + v3_z)
    tau23 = v2_z + v3_y
    tau33 = 4.0 / 3.0 * v3_z - 2.0 / 3.0 * (v1_x + v2_y)

    tau11_x = 4.0 / 3.0 * v1_xx - 2.0 / 3.0 * (v2_xy + v3_xz)
    tau12_x = v1_xy + v2_xx
    tau13_x = v1_xz + v3_xx

    tau12_y = v1_yy + v2_xy
    tau22_y = 4.0 / 3.0 * v2_yy - 2.0 / 3.0 * (v1_xy + v3_yz)
    tau23_y = v2_yz + v3_yy

    tau13_z = v1_zz + v3_xz
    tau23_z = v2_zz + v3_yz
    tau33_z = 4.0 / 3.0 * v3_zz - 2.0 / 3.0 * (v1_xz + v2_yz)

    # Compute the source terms
    # Density equation
    du1 = (rho_t + rho_x * v1 + rho * v1_x
           + rho_y * v2 + rho * v2_y
           + rho_z * v3 + rho * v3_z)
    # x-momentum equation
    du2 = (rho_t * v1 + rho * v1_t + p_x + rho_x * v1^2
           + 2.0 * rho * v1 * v1_x
           + rho_y * v1 * v2
           + rho * v1_y * v2
           + rho * v1 * v2_y
           + rho_z * v1 * v3
           + rho * v1_z * v3
           + rho * v1 * v3_z -
           mu_ * (tau11_x + tau12_y + tau13_z))
    # y-momentum equation
    du3 = (rho_t * v2 + rho * v2_t + p_y + rho_x * v1 * v2
           + rho * v1_x * v2
           + rho * v1 * v2_x
           + rho_y * v2^2
           + 2.0 * rho * v2 * v2_y
           + rho_z * v2 * v3
           + rho * v2_z * v3
           + rho * v2 * v3_z -
           mu_ * (tau12_x + tau22_y + tau23_z))
    # z-momentum equation
    du4 = (rho_t * v3 + rho * v3_t + p_z + rho_x * v1 * v3
           + rho * v1_x * v3
           + rho * v1 * v3_x
           + rho_y * v2 * v3
           + rho * v2_y * v3
           + rho * v2 * v3_y
           + rho_z * v3^2
           + 2.0 * rho * v3 * v3_z -
           mu_ * (tau13_x + tau23_y + tau33_z))
    # Total energy equation
    du5 = (E_t + v1_x * (E + p) + v1 * (E_x + p_x)
           + v2_y * (E + p) + v2 * (E_y + p_y)
           + v3_z * (E + p) + v3 * (E_z + p_z) -
           # stress tensor and temperature gradient from x-direction
           mu_ * (q_xx + v1_x * tau11 + v2_x * tau12 + v3_x * tau13
            + v1 * tau11_x + v2 * tau12_x + v3 * tau13_x) -
           # stress tensor and temperature gradient terms from y-direction
           mu_ * (q_yy + v1_y * tau12 + v2_y * tau22 + v3_y * tau23
            + v1 * tau12_y + v2 * tau22_y + v3 * tau23_y) -
           # stress tensor and temperature gradient terms from z-direction
           mu_ * (q_zz + v1_z * tau13 + v2_z * tau23 + v3_z * tau33
            + v1 * tau13_z + v2 * tau23_z + v3 * tau33_z))

    return SVector(du1, du2, du3, du4, du5)
end

initial_condition = initial_condition_navier_stokes_convergence_test

# BC types
velocity_bc_top_bottom = NoSlip() do x, t, equations
    u = initial_condition_navier_stokes_convergence_test(x, t, equations)
    return SVector(u[2], u[3], u[4])
end
heat_bc_top_bottom = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_top_bottom = BoundaryConditionNavierStokesWall(velocity_bc_top_bottom,
                                                                  heat_bc_top_bottom)

# define inviscid boundary conditions
boundary_conditions = (; x_neg = boundary_condition_periodic,
                       x_pos = boundary_condition_periodic,
                       y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall,
                       z_neg = boundary_condition_periodic,
                       z_pos = boundary_condition_periodic)

# define viscous boundary conditions
boundary_conditions_parabolic = (; x_neg = boundary_condition_periodic,
                                 x_pos = boundary_condition_periodic,
                                 y_neg = boundary_condition_top_bottom,
                                 y_pos = boundary_condition_top_bottom,
                                 z_neg = boundary_condition_periodic,
                                 z_pos = boundary_condition_periodic)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic),
                                             source_terms = source_terms_navier_stokes_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol, dt = 1e-5,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary
