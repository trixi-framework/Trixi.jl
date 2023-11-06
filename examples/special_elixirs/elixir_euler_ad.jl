
# This example is described in more detail in the documentation of Trixi.jl

using Trixi, LinearAlgebra, ForwardDiff

equations = CompressibleEulerEquations2D(1.4)

mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0),
                initial_refinement_level = 2, n_cells_max = 10^5)

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
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
    # needs appropriate mesh size, e.g. [-10,-10]x[10,10]
    # for error convergence: make sure that the end time is such that the vortex is back at the initial state!!
    # for the current velocity and domain size: t_end should be a multiple of 20s
    # initial center of the vortex
    inicenter = SVector(0.0, 0.0)
    # size and strength of the vortex
    iniamplitude = 5.0
    # base flow
    rho = 1.0
    v1 = 1.0
    v2 = 1.0
    vel = SVector(v1, v2)
    p = 25.0
    rt = p / rho                  # ideal gas equation
    t_loc = 0.0
    cent = inicenter + vel * t_loc      # advection of center
    # ATTENTION: handle periodic BC, but only for v1 = v2 = 1.0 (!!!!)
    cent = x - cent # distance to center point
    # cent = cross(iniaxis, cent) # distance to axis, tangent vector, length r
    # cross product with iniaxis = [0, 0, 1]
    cent = SVector(-cent[2], cent[1])
    r2 = cent[1]^2 + cent[2]^2
    du = iniamplitude / (2 * π) * exp(0.5 * (1 - r2)) # vel. perturbation
    dtemp = -(equations.gamma - 1) / (2 * equations.gamma * rt) * du^2 # isentropic
    rho = rho * (1 + dtemp)^(1 / (equations.gamma - 1))
    vel = vel + du * cent
    v1, v2 = vel
    p = p * (1 + dtemp)^(equations.gamma / (equations.gamma - 1))
    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_isentropic_vortex,
                                    solver)

u0_ode = compute_coefficients(0.0, semi)

J = ForwardDiff.jacobian((du_ode, γ) -> begin
                             equations_inner = CompressibleEulerEquations2D(first(γ))
                             semi_inner = Trixi.remake(semi, equations = equations_inner,
                                                       uEltype = eltype(γ))
                             Trixi.rhs!(du_ode, u0_ode, semi_inner, 0.0)
                         end, similar(u0_ode), [1.4]); # γ needs to be an `AbstractArray`
