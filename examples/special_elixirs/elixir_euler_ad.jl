# This example is described in more detail in the documentation of Trixi.jl

using Trixi, LinearAlgebra, ForwardDiff

equations = CompressibleEulerEquations2D(1.4)

mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0),
                initial_refinement_level = 2, n_cells_max = 10^5, periodicity = true)

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
# `StepsizeCallback` (CFL-Condition) and less diffusion.
solver = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed_naive),
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

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
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_isentropic_vortex,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

u0_ode = compute_coefficients(0.0, semi)

J = ForwardDiff.jacobian((du_ode, γ) -> begin
                             equations_inner = CompressibleEulerEquations2D(first(γ))
                             semi_inner = Trixi.remake(semi, equations = equations_inner,
                                                       uEltype = eltype(γ))
                             Trixi.rhs!(du_ode, u0_ode, semi_inner, 0.0)
                         end, similar(u0_ode), [1.4]); # γ needs to be an `AbstractArray`
