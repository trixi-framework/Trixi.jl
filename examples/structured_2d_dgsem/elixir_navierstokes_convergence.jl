using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.72
mu() = 0.01

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

# Create DG solver with polynomial degree = 3 and local Lax-Friedrichs flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())

# Mapping as described in https://arxiv.org/abs/2012.12040, but reduced to 2D.
# Maps reference coordinates (xi_, eta_) in [-1,1]^2 to physical domain ~[0,3]^2
# with a wavy (non-Cartesian) deformation. Opposite edges match, so the mesh is periodic.
function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0,3]
    xi = 1.5 * xi_ + 1.5
    eta = 1.5 * eta_ + 1.5

    y = eta + 3 / 8 * (cos(1.5 * pi * (2 * xi - 3) / 3) *
                       cos(0.5 * pi * (2 * eta - 3) / 3))

    x = xi + 3 / 8 * (cos(0.5 * pi * (2 * xi - 3) / 3) *
                      cos(2 * pi * (2 * y - 3) / 3))

    return SVector(x, y)
end

cells_per_dimension = (8, 8)

mesh = StructuredMesh(cells_per_dimension, mapping, periodicity = true)

# The manufactured solution is periodic on [0,3] x [0,3] with wavenumber k = 2π/3.
# The velocity field is divergence-free: ∂v1/∂x + ∂v2/∂y = 0, which simplifies
# the continuity source term. 
function initial_condition_navier_stokes_convergence_test(x, t, equations)
    RealT = eltype(x)
    A = 0.5f0
    c = 2
    k = 2 * convert(RealT, pi) / 3  # wavenumber: period = 3
    kx = k * x[1]
    ky = k * x[2]
    pt = convert(RealT, pi) * t

    rho = c + A * sin(kx) * cos(ky) * cos(pt)
    v1 = A * sin(kx) * cos(ky) * cos(pt)
    v2 = -A * cos(kx) * sin(ky) * cos(pt)  # div-free: ∂v1/∂x + ∂v2/∂y = 0
    p = rho^2

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

@inline function source_terms_navier_stokes_convergence_test(u, x, t, equations)
    RealT = eltype(x)

    # TODO: parabolic
    # we currently need to hardcode these parameters until we fix the "combined equation" issue
    # see also https://github.com/trixi-framework/Trixi.jl/pull/1160
    inv_gamma_minus_one = inv(equations.gamma - 1)
    Pr = prandtl_number()
    mu_ = mu()

    # Same settings as in `initial_condition`
    A = 0.5f0
    c = 2
    k = 2 * convert(RealT, pi) / 3  # wavenumber: period = 3
    k2 = k * k

    kx = k * x[1]
    ky = k * x[2]
    pt = convert(RealT, pi) * t

    # Shorthand trig values
    Sx = sin(kx)
    Cx = cos(kx)
    Sy = sin(ky)
    Cy = cos(ky)
    Ct = cos(pt)
    St = sin(pt)

    # Manufactured solution and all necessary derivatives
    rho = c + A * Sx * Cy * Ct
    rho_t = -convert(RealT, pi) * A * Sx * Cy * St
    rho_x = A * k * Cx * Cy * Ct
    rho_y = -A * k * Sx * Sy * Ct
    rho_xx = -A * k2 * Sx * Cy * Ct
    rho_yy = -A * k2 * Sx * Cy * Ct  # same expression as rho_xx

    v1 = A * Sx * Cy * Ct
    v1_t = -convert(RealT, pi) * A * Sx * Cy * St
    v1_x = A * k * Cx * Cy * Ct
    v1_y = -A * k * Sx * Sy * Ct
    v1_xx = -A * k2 * Sx * Cy * Ct
    v1_xy = -A * k2 * Cx * Sy * Ct
    v1_yy = -A * k2 * Sx * Cy * Ct

    v2 = -A * Cx * Sy * Ct
    v2_t = convert(RealT, pi) * A * Cx * Sy * St
    v2_x = A * k * Sx * Sy * Ct
    v2_y = -A * k * Cx * Cy * Ct  # = -v1_x, confirming div-free
    v2_xx = A * k2 * Cx * Sy * Ct
    v2_xy = A * k2 * Sx * Cy * Ct
    v2_yy = A * k2 * Cx * Sy * Ct

    p = rho * rho
    p_t = 2 * rho * rho_t
    p_x = 2 * rho * rho_x
    p_y = 2 * rho * rho_y
    p_xx = 2 * (rho * rho_xx + rho_x * rho_x)
    p_yy = 2 * (rho * rho_yy + rho_y * rho_y)

    E = p * inv_gamma_minus_one + 0.5f0 * rho * (v1^2 + v2^2)
    E_t = (p_t * inv_gamma_minus_one + rho_t * 0.5f0 * (v1^2 + v2^2) +
           rho * (v1 * v1_t + v2 * v2_t))
    E_x = (p_x * inv_gamma_minus_one + rho_x * 0.5f0 * (v1^2 + v2^2) +
           rho * (v1 * v1_x + v2 * v2_x))
    E_y = (p_y * inv_gamma_minus_one + rho_y * 0.5f0 * (v1^2 + v2^2) +
           rho * (v1 * v1_y + v2 * v2_y))

    # Some convenience constants
    T_const = equations.gamma * inv_gamma_minus_one / Pr
    inv_rho_cubed = 1 / (rho^3)

    # compute the source terms
    # density equation
    du1 = rho_t + rho_x * v1 + rho * v1_x + rho_y * v2 + rho * v2_y

    # x-momentum equation
    du2 = (rho_t * v1 + rho * v1_t + p_x + rho_x * v1^2
           + 2 * rho * v1 * v1_x
           + rho_y * v1 * v2
           + rho * v1_y * v2
           + rho * v1 * v2_y -
           # stress tensor from x-direction
           RealT(4) / 3 * v1_xx * mu_ +
           RealT(2) / 3 * v2_xy * mu_ -
           v1_yy * mu_ -
           v2_xy * mu_)
    # y-momentum equation
    du3 = (rho_t * v2 + rho * v2_t + p_y + rho_x * v1 * v2
           + rho * v1_x * v2
           + rho * v1 * v2_x
           + rho_y * v2^2
           + 2 * rho * v2 * v2_y -
           # stress tensor from y-direction
           v1_xy * mu_ -
           v2_xx * mu_ -
           RealT(4) / 3 * v2_yy * mu_ +
           RealT(2) / 3 * v1_xy * mu_)
    # total energy equation
    du4 = (E_t + v1_x * (E + p) + v1 * (E_x + p_x)
           + v2_y * (E + p) + v2 * (E_y + p_y) -
           # stress tensor and temperature gradient terms from x-direction
           RealT(4) / 3 * v1_xx * v1 * mu_ +
           RealT(2) / 3 * v2_xy * v1 * mu_ -
           RealT(4) / 3 * v1_x * v1_x * mu_ +
           RealT(2) / 3 * v2_y * v1_x * mu_ -
           v1_xy * v2 * mu_ -
           v2_xx * v2 * mu_ -
           v1_y * v2_x * mu_ -
           v2_x * v2_x * mu_ -
           T_const * inv_rho_cubed *
           (p_xx * rho * rho -
            2 * p_x * rho * rho_x +
            2 * p * rho_x * rho_x -
            p * rho * rho_xx) * mu_ -
           # stress tensor and temperature gradient terms from y-direction
           v1_yy * v1 * mu_ -
           v2_xy * v1 * mu_ -
           v1_y * v1_y * mu_ -
           v2_x * v1_y * mu_ -
           RealT(4) / 3 * v2_yy * v2 * mu_ +
           RealT(2) / 3 * v1_xy * v2 * mu_ -
           RealT(4) / 3 * v2_y * v2_y * mu_ +
           RealT(2) / 3 * v1_x * v2_y * mu_ -
           T_const * inv_rho_cubed *
           (p_yy * rho * rho -
            2 * p_y * rho * rho_y +
            2 * p * rho_y * rho_y -
            p * rho * rho_yy) * mu_)

    return SVector(du1, du2, du3, du4)
end

initial_condition = initial_condition_navier_stokes_convergence_test

# Fully periodic boundary conditions (mesh periodicity matches the manufactured solution)
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic),
                                             source_terms = source_terms_navier_stokes_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.5)
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
