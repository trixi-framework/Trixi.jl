using OrdinaryDiffEqLowStorageRK
using Trixi

dg = DGMulti(polydeg = 3, element_type = Line(), approximation_type = GaussSBP(),
             surface_integral = SurfaceIntegralWeakForm(flux_hllc))

prandtl_number() = 0.72
mu() = 6.25e-4 # equivalent to Re = 1600

equations = CompressibleEulerEquations1D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

function initial_condition_navier_stokes_convergence_test(x, t, equations)
    # Amplitude and shift
    RealT = eltype(x)
    A = 0.5f0
    c = 2

    # convenience values for trig. functions
    pi_x = convert(RealT, pi) * x[1]
    pi_t = convert(RealT, pi) * t

    rho = c + A * sin(pi_x) * cos(pi_t)
    v1 = sin(pi_x) * cos(pi_t)
    p = rho^2

    return prim2cons(SVector(rho, v1, p), equations)
end
initial_condition = initial_condition_navier_stokes_convergence_test

@inline function source_terms_navier_stokes_convergence_test(u, x, t, equations)
    # we currently need to hardcode these parameters until we fix the "combined equation" issue
    # see also https://github.com/trixi-framework/Trixi.jl/pull/1160
    RealT = eltype(x)
    inv_gamma_minus_one = inv(equations.gamma - 1)
    Pr = prandtl_number()
    mu_ = mu()

    # Same settings as in `initial_condition`
    # Amplitude and shift
    A = 0.5f0
    c = 2

    # convenience values for trig. functions
    pi_x = convert(RealT, pi) * x[1]
    pi_t = convert(RealT, pi) * t

    # compute the manufactured solution and all necessary derivatives
    rho = c + A * sin(pi_x) * cos(pi_t)
    rho_t = -convert(RealT, pi) * A * sin(pi_x) * sin(pi_t)
    rho_x = convert(RealT, pi) * A * cos(pi_x) * cos(pi_t)
    rho_xx = -convert(RealT, pi) * convert(RealT, pi) * A * sin(pi_x) * cos(pi_t)

    v1 = sin(pi_x) * cos(pi_t)
    v1_t = -convert(RealT, pi) * sin(pi_x) * sin(pi_t)
    v1_x = convert(RealT, pi) * cos(pi_x) * cos(pi_t)
    v1_xx = -convert(RealT, pi) * convert(RealT, pi) * sin(pi_x) * cos(pi_t)

    p = rho * rho
    p_t = 2 * rho * rho_t
    p_x = 2 * rho * rho_x
    p_xx = 2 * rho * rho_xx + 2 * rho_x * rho_x

    E = p * inv_gamma_minus_one + 0.5f0 * rho * v1^2
    E_t = p_t * inv_gamma_minus_one + 0.5f0 * rho_t * v1^2 + rho * v1 * v1_t
    E_x = p_x * inv_gamma_minus_one + 0.5f0 * rho_x * v1^2 + rho * v1 * v1_x

    # Some convenience constants
    T_const = equations.gamma * inv_gamma_minus_one / Pr
    inv_rho_cubed = 1 / (rho^3)

    # compute the source terms
    # density equation
    du1 = rho_t + rho_x * v1 + rho * v1_x

    # x-momentum equation
    du2 = (rho_t * v1 + rho * v1_t
           + p_x + rho_x * v1^2 + 2 * rho * v1 * v1_x -
           # stress tensor from x-direction
           v1_xx * mu_)

    # total energy equation
    du3 = (E_t + v1_x * (E + p) + v1 * (E_x + p_x) -
           # stress tensor and temperature gradient terms from x-direction
           v1_xx * v1 * mu_ -
           v1_x * v1_x * mu_ -
           T_const * inv_rho_cubed *
           (p_xx * rho * rho -
            2 * p_x * rho * rho_x +
            2 * p * rho_x * rho_x -
            p * rho * rho_xx) * mu_)

    return SVector(du1, du2, du3)
end

cells_per_dimension = (12,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-1.0,), coordinates_max = (1.0,),
                   periodicity = true)

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic),
                                             source_terms = source_terms_navier_stokes_convergence_test)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

alive_callback = AliveCallback(alive_interval = 10)

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))

callbacks = CallbackSet(summary_callback, alive_callback,
                        analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
