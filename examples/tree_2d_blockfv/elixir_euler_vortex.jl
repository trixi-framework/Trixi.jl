using OrdinaryDiffEqLowOrderRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case of
- Chi-Wang Shu (1997)
  Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory
  Schemes for Hyperbolic Conservation Laws
  [NASA/CR-97-206253](https://ntrs.nasa.gov/citations/19980007543)
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
    # This needs appropriate mesh size to ensure that the perturbation is
    # essentially zero to machine accuracy at the boundary so that we can
    # use periodic boundary conditions.
    # In addition, we assume that the domain is of size [-10, -10] x [10, 10]
    # to handle periodic boundary conditions correctly below.

    # initial center of the vortex
    RealT = eltype(x)
    center = SVector(0, 0)

    # size and strength of the vortex
    amplitude = 5

    # base flow
    rho = 1
    v1 = 1
    v2 = 1
    vel = SVector(v1, v2)
    p = convert(RealT, 25)

    rt = p / rho             # R*T by ideal gas equation
    cent = center + vel * t  # advection of center
    cent = x - cent # distance to center point

    # The domain [-10, 10]^2 is periodic with edge length 20 in each direction.
    # To obtain a valid initial condition for any time `t` (and any base
    # velocity `vel`), wrap the distance vector to the nearest periodic image
    # of the advected vortex center.
    domain_length = convert(RealT, 20)
    cent = cent - domain_length * round.(cent / domain_length)

    # cent = cross(iniaxis, cent) # distance to axis, tangent vector, length r
    # cross product with iniaxis = [0, 0, 1]
    cent = SVector(-cent[2], cent[1])

    r2 = cent[1]^2 + cent[2]^2
    du = amplitude / (2 * convert(RealT, pi)) * exp(0.5f0 * (1 - r2)) # vel. perturbation
    dtemp = -(equations.gamma - 1) / (2 * equations.gamma * rt) * du^2 # isentropic
    rho = rho * (1 + dtemp)^(1 / (equations.gamma - 1))
    vel = vel + du * cent
    v1, v2 = vel
    p = p * (1 + dtemp)^(equations.gamma / (equations.gamma - 1))

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

solver = BlockFV(n_nodes = 10, surface_flux = flux_hllc)

coordinates_min = (-10.0, -10.0)
coordinates_max = (10.0, 10.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations,
                                    initial_condition_isentropic_vortex,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback,
                        alive_callback)

###############################################################################
# run the simulation
sol = solve(ode, Euler();
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
