using OrdinaryDiffEqLowStorageRK
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
                                                               slope = 1000)
    if x[2] < 0.5
        p = 2 * x[2] + 1
    else
        p = x[2] + 3 / 2
    end

    # smooth the discontinuity to avoid ambiguity at element interfaces
    smoothed_heaviside(x, left, right) = left + 0.5 * (1 + tanh(slope * x)) * (right - left)
    rho = smoothed_heaviside(x[2] - 0.5, 2.0, 1.0)

    c = sqrt(equations.gamma * p / rho)
    # the velocity is multiplied by sin(pi*y)^6 as in Remacle et al. 2003 to ensure that the
    # initial condition satisfies reflective boundary conditions at the top/bottom boundaries.
    v = -0.025 * c * cos(8 * pi * x[1]) * sin(pi * x[2])^6
    u = 0.0

    return prim2cons(SVector(rho, u, v, p), equations)
end
initial_condition = initial_condition_rayleigh_taylor_instability

@inline function source_terms_rayleigh_taylor_instability(u, x, t,
                                                          equations::CompressibleEulerEquations2D)
    g = 1.0
    rho, _, rho_v2, _ = u

    return SVector(0.0, 0.0, g * rho, g * rho_v2)
end

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
surface_flux = flux_hll
volume_flux = flux_ranocha
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.01, # Relatively large value, i.e., limit late!
                                         alpha_smooth = true,
                                         variable = density)

volume_integral = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                  volume_flux_dg = volume_flux,
                                                  volume_flux_fv = surface_flux,
                                                  slope_limiter = monotonized_central)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (0.0, 0.0) # (min(x), min(y))
coordinates_max = (0.25, 1.0) # (max(x), max(y))

num_elements_per_dimension = 2
trees_per_dimension = (num_elements_per_dimension, num_elements_per_dimension * 4)

mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 0,
                 periodicity = (true, false))

# Setup: left/right periodic BCs and Dirichlet BCs on the top/bottom.

# Bottom boundary causes for standard Dirichlet BC some spurious oscillations
# and thus spurious mesh refinement.
# To prevent this, we implement a special subsonic boundary condition that
# enforces the pressure from the initial condition but takes all other variables
# from the interior solution.
#
# See Section 2.3 of the reference below for a discussion of robust
# subsonic inflow/outflow boundary conditions.
#
# - Jan-Reneé Carlson (2011)
#   Inflow/Outflow Boundary Conditions with Application to FUN3D.
#   [NASA TM 20110022658](https://ntrs.nasa.gov/citations/20110022658)
@inline function boundary_condition_subsonic(u_inner,
                                             normal_direction::AbstractVector,
                                             x, t, surface_flux_function,
                                             equations::CompressibleEulerEquations2D)
    rho_loc, v1_loc, v2_loc, p_loc = cons2prim(u_inner, equations)

    # For subsonic boundary: Take pressure from initial condition
    p_loc = pressure(initial_condition_rayleigh_taylor_instability(x, t, equations),
                     equations)

    prim = SVector(rho_loc, v1_loc, v2_loc, p_loc)
    u_surface = prim2cons(prim, equations)

    return flux(u_surface, normal_direction, equations)
end

boundary_conditions = Dict(:y_neg => boundary_condition_subsonic,
                           :y_pos => BoundaryConditionDirichlet(initial_condition_rayleigh_taylor_instability))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_terms_rayleigh_taylor_instability,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      max_level = 5, max_threshold = 0.1)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 10,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        amr_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(thread = Trixi.True());
            abstol = time_int_tol, reltol = time_int_tol,
            adaptive = true, dt = 1e-3, # needed only for tests/CI
            ode_default_options()..., callback = callbacks);
