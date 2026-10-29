using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqCore: PIDController
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

# Turbulent plane channel flow at a friction Reynolds number Re_tau = 180.
# The mean velocity profile and the Reynolds stresses can be compared with the
# incompressible DNS data of
# - R. D. Moser, J. Kim, N. N. Mansour (1999)
#   Direct numerical simulation of turbulent channel flow up to Re_tau = 590
#   [DOI: 10.1063/1.869966](https://doi.org/10.1063/1.869966)
#
# The domain is (0, 2pi) x (-1, 1) x (0, pi) with periodic boundaries in the
# streamwise (x) and spanwise (z) directions and no-slip isothermal walls at y = ±1.
# The channel half-height is delta = 1 and the density is rho ≈ 1. In the statistically
# steady state, the wall shear stress balances the constant pressure gradient
# dp/dx = -1, i.e., tau_w = -dp/dx * delta = 1. Thus, the friction velocity is
# u_tau = sqrt(tau_w / rho) = 1 and Re_tau = rho * u_tau * delta / mu = 1 / mu.

reynolds_number_tau() = 180.0
gamma() = 1.4
prandtl_number() = 0.72
mach_number_bulk() = 0.1 # low Mach number, i.e., nearly incompressible flow
mu() = 1 / reynolds_number_tau()

equations = CompressibleEulerEquations3D(gamma())
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

# The bulk Reynolds number Re_b = 2 * delta * v_bulk / nu = 5600 of the DNS of
# - J. Kim, P. Moin, R. Moser (1987)
#   Turbulence statistics in fully developed channel flow at low Reynolds number
#   [DOI: 10.1017/S0022112087000892](https://doi.org/10.1017/S0022112087000892)
# yields the bulk velocity in wall units. It is only used to set the background
# pressure for the prescribed bulk Mach number.
velocity_bulk() = 5600 / (2 * reynolds_number_tau())

# uniform initial density and pressure
rho_reference() = 1.0
p_reference() = (velocity_bulk() / mach_number_bulk())^2 * rho_reference() / gamma()
# Wall temperature T = p / (R * rho) consistent with the reference state
function temperature_wall(equations_parabolic)
    return p_reference() / (equations_parabolic.R * rho_reference())
end

"""
    initial_condition_channel_flow(x, t, equations::CompressibleEulerEquations3D)

Mean turbulent velocity profile superimposed with oblique waves with random wave numbers
and phases.
Since the laminar flow at this Reynolds number is linearly stable, the transition to
turbulence requires a perturbation of finite amplitude. Thus, the amplitude is chosen
such that the velocity fluctuations are of the order of the turbulent ones of the
DNS of
- R. D. Moser, J. Kim, N. N. Mansour (1999)
  Direct numerical simulation of turbulent channel flow up to Re_tau = 590
  [DOI: 10.1063/1.869966](https://doi.org/10.1063/1.869966)

Each wave is divergence-free, periodic, and vanishes at the walls.
"""
function initial_condition_channel_flow(x, t, equations::CompressibleEulerEquations3D)
    # mean velocity profile in wall units with the distance to the closest wall y_plus
    y_plus = (1 - abs(x[2])) * reynolds_number_tau()
    v1 = 1 / 0.41 * log(1 + 0.41 * y_plus) +
         7.8 * (1 - exp(-y_plus / 11) - y_plus / 11 * exp(-y_plus / 3))
    v2 = 0.0
    v3 = 0.0

    # Randomly chosen parameters (kx, kz, m, phase) of the waves with
    # - the streamwise wave number kx in 1:12, periodic in (0, 2pi)
    # - the spanwise wave number kz in 2 * (2:13), periodic in (0, pi)
    # - the number of half waves between the walls m in 4:16
    # - the phase in (0, 2pi)
    waves = ((2, 4, 15, 2.9881439602198525),
             (8, 10, 15, 3.5405555223847127),
             (5, 10, 8, 3.405022085800654),
             (8, 12, 6, 5.346064788384369),
             (3, 16, 9, 3.238623650250496),
             (2, 20, 16, 5.2708623318249295),
             (11, 22, 8, 4.650444774080629),
             (9, 26, 10, 1.572832553058131),
             (5, 22, 7, 3.9922177526927602),
             (3, 10, 15, 2.5315774971239198),
             (3, 26, 6, 6.095619071195619),
             (1, 6, 14, 4.217091595120723),
             (6, 20, 15, 1.1991564505511239),
             (6, 8, 10, 2.212437973608053),
             (3, 26, 11, 3.057180229064999),
             (10, 22, 6, 1.9759887859013716))
    amplitude = 0.65
    for (kx, kz, m, phase) in waves
        sin_theta, cos_theta = sincos(kx * x[1] + kz * x[3] + phase)
        g = amplitude * sin(0.5 * m * pi * (1 + x[2]))^2
        dg = amplitude * 0.5 * m * pi * sin(m * pi * (1 + x[2]))
        k_squared = kx^2 + kz^2

        v1 -= kx / k_squared * dg * sin_theta
        v2 += g * cos_theta
        v3 -= kz / k_squared * dg * sin_theta
    end

    return prim2cons(SVector(rho_reference(), v1, v2, v3, p_reference()), equations)
end
initial_condition = initial_condition_channel_flow

# Constant streamwise pressure gradient dp/dx = -1 driving the flow, implemented as a
# body force. The energy equation contains the work of this force per unit time.
@inline function source_terms_channel_flow(u, x, t, equations::CompressibleEulerEquations3D)
    rho, rho_v1, _, _, _ = u
    dpdx = -1
    v1 = rho_v1 / rho

    return SVector(0, -dpdx, 0, 0, -dpdx * v1)
end

# `flux_shima_etal_turbo` is equivalent to `flux_shima_etal` but uses a specialized
# volume integral kernel with SIMD instructions.
polydeg = 5
surface_flux = flux_lax_friedrichs
volume_flux = flux_shima_etal_turbo
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

# The trees are clustered towards the walls by a tanh stretching in y-direction.
# The number of trees in x- and z-direction is proportional to the domain size to
# obtain the same resolution in streamwise and spanwise direction.
function mapping(xi, eta, zeta)
    stretching = 2.4

    x = pi * (xi + 1)
    y = tanh(stretching * eta) / tanh(stretching)
    z = 0.5 * pi * (zeta + 1)

    return SVector(x, y, z)
end

trees_per_dimension = (16, 8, 8)
mesh = P4estMesh(trees_per_dimension, polydeg = polydeg, mapping = mapping,
                 periodicity = (true, false, true), initial_refinement_level = 0)

# The no-slip and isothermal wall conditions are imposed via the viscous fluxes.
# The hyperbolic part uses slip walls.
boundary_conditions = (; y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall)

velocity_bc_wall = NoSlip((x, t, equations_parabolic) -> SVector(0.0, 0.0, 0.0))
heat_bc_wall = Isothermal((x, t, equations_parabolic) -> temperature_wall(equations_parabolic))
boundary_condition_wall = BoundaryConditionNavierStokesWall(velocity_bc_wall, heat_bc_wall)
boundary_conditions_parabolic = (; y_neg = boundary_condition_wall,
                                 y_pos = boundary_condition_wall)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic),
                                             source_terms = source_terms_channel_flow)

###############################################################################
# ODE solvers, callbacks etc.

# The time is measured in units of delta / u_tau. Use a much longer final time to
# collect turbulence statistics.
tspan = (0.0, 15.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (energy_kinetic,
                                                                 enstrophy))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 5000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution)

###############################################################################
# run the simulation

time_int_tol = 1e-6
sol = solve(ode, SSPRK43(thread = Trixi.Threaded());
            controller = PIDController(0.55, -0.27, 0.05), # optimized for SSPRK43
            dt = 1e-6, abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
