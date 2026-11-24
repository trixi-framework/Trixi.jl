using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations: Sod shock tube

equations = CompressibleEulerEquations1D(1.4)

"""
    initial_condition_sod(x, t, equations::CompressibleEulerEquations1D)

Shock tube of
- Gary A. Sod (1978)
  A survey of several finite difference methods for systems of nonlinear
  hyperbolic conservation laws
  [DOI: 10.1016/0021-9991(78)90023-2](https://doi.org/10.1016/0021-9991(78)90023-2)
with the left state `(rho, v, p) = (1, 0, 1)` and the right state
`(0.125, 0, 0.1)`, separated at `x = 0.5`.
"""
function initial_condition_sod(x, t, equations::CompressibleEulerEquations1D)
    RealT = eltype(x)
    if x[1] < 0.5f0
        return prim2cons(SVector{3, RealT}(1, 0, 1), equations)
    else
        return prim2cons(SVector{3, RealT}(0.125f0, 0, convert(RealT, 0.1)), equations)
    end
end
initial_condition = initial_condition_sod

# Macro-cell finite volume solver with TVD-limited MUSCL reconstruction of the
# primitive variables
n_cells = 8
slope_limiter = minmod
surface_flux = flux_lax_friedrichs
solver = MacroCellFV(n_cells = n_cells, surface_flux = surface_flux,
                     reconstruction = ReconstructionMUSCL(slope_limiter),
                     center2recon = cons2prim, face2flux = prim2cons)

coordinates_min = 0.0
coordinates_max = 1.0
initial_refinement_level = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = false)

# Transmissive boundaries: the ghost subcells mirror the interior subcells
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_do_nothing)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK33();
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
