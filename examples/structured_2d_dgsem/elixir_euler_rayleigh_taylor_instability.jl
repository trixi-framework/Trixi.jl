
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_rayleigh_taylor_instability(coordinates, t, equations::CompressibleEulerEquations2D)

Setup used for the Rayleigh-Taylor instability. Initial condition adapted from
- Shi, Jing, Yong-Tao Zhang, and Chi-Wang Shu (2003).
  Resolution of high order WENO schemes for complicated flow structures.
  [DOI](https://doi.org/10.1016/S0021-9991(03)00094-9).
- Remacle, Jean-François, Joseph E. Flaherty, and Mark S. Shephard (2003).
  An adaptive discontinuous Galerkin technique with an orthogonal basis applied to compressible
  flow problems.
  [DOI](https://doi.org/10.1137/S00361445023830)

The domain is [0, 0.25] x [0, 1]. Boundary conditions can be reflective wall boundary conditions on
all boundaries or
- periodic boundary conditions on the left/right boundaries
- Dirichlet boundary conditions on the top/bottom boundaries

This should be used together with `source_terms_rayleigh_taylor_instability`, which is
defined below.
"""
@inline function initial_condition_rayleigh_taylor_instability(x, t,
                                                               equations::CompressibleEulerEquations2D,
                                                               slope=1000)
  tol = 1e2*eps()

  if x[2] < 0.5
    p = 2*x[2] + 1
  else
    p = x[2] + 3/2
  end

  # smooth the discontinuity to avoid ambiguity at element interfaces
  smoothed_heaviside(x, left, right) = left + 0.5*(1 + tanh(slope * x)) * (right-left)
  rho = smoothed_heaviside(x[2] - 0.5, 2.0, 1.0)

  c = sqrt(equations.gamma * p / rho)
  # the velocity is multiplied by sin(pi*y)^6 as in Remacle et al. 2003 to ensure that the
  # initial condition satisfies reflective boundary conditions at the top/bottom boundaries.
  v = -0.025 * c * cos(8*pi*x[1]) * sin(pi*x[2])^6
  u = 0.0

  return prim2cons(SVector(rho, u, v, p), equations)
end

@inline function source_terms_rayleigh_taylor_instability(u, x, t,
                                                          equations::CompressibleEulerEquations2D)
  g = 1.0
  rho, rho_v1, rho_v2, rho_e = u

  return SVector(0.0, 0.0, g*rho, g*rho_v2)
end

# numerical parameters
volume_flux = flux_ranocha
solver = DGSEM(polydeg=3, surface_flux=flux_hll,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

# The domain is [0, 0.25] x [0, 1]
mapping(xi, eta) = SVector(0.25 * 0.5 * (1.0 + xi), 0.5 * (1.0 + eta))

num_elements_per_dimension = 32
cells_per_dimension = (num_elements_per_dimension, num_elements_per_dimension * 4)
mesh = StructuredMesh(cells_per_dimension, mapping)

initial_condition = initial_condition_rayleigh_taylor_instability
boundary_conditions = (
                       x_neg=boundary_condition_slip_wall,
                       x_pos=boundary_condition_slip_wall,
                       y_neg=boundary_condition_slip_wall,
                       y_pos=boundary_condition_slip_wall,
                      )

# # Alternative setup: left/right periodic BCs and Dirichlet BCs on the top/bottom.
# boundary_conditions = (
#                        x_neg=boundary_condition_periodic,
#                        x_pos=boundary_condition_periodic,
#                        y_neg=BoundaryConditionDirichlet(initial_condition),
#                        y_pos=BoundaryConditionDirichlet(initial_condition),
#                       )

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms=source_terms_rayleigh_taylor_instability,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.65)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callbacks);

summary_callback() # print the timer summary
