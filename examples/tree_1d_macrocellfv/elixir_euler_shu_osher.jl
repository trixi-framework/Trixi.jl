using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

# Shu-Osher initial condition for 1D compressible Euler equations
# Example 8 from Shu, Osher (1989).
# https://doi.org/10.1016/0021-9991(89)90222-2
function initial_condition_shu_osher(x, t, equations::CompressibleEulerEquations1D)
    RealT = eltype(x)
    x0 = -4

    rho_left = convert(RealT, 27 / 7)
    v_left = 4 * sqrt(convert(RealT, 35)) / 9
    p_left = convert(RealT, 31 / 3)

    v_right = zero(RealT)
    p_right = one(RealT)

    rho = ifelse(x[1] > x0, 1 + sin(5 * x[1]) / 5, rho_left)
    v = ifelse(x[1] > x0, v_right, v_left)
    p = ifelse(x[1] > x0, p_right, p_left)

    return prim2cons(SVector(rho, v, p), equations)
end
initial_condition = initial_condition_shu_osher

# Macro-cell finite volume solver with WENO5-Z reconstruction of the conservative
# variables from the subcell averages. The initial condition is set to the subcell
# averages (Gauss-Legendre quadrature with 5 nodes per subcell).
# The primitive variables can be reconstructed instead with `center2recon = cons2prim`
# and `face2flux = prim2cons`. From the subcell averages (`MomentInversionIdentity`),
# this is only second-order accurate for smooth solutions, since `cons2prim` acts on the
# averages. With `moment_inversion = MomentInversionWENO()`, `cons2prim` acts on the
# point values at the subcell centers (fourth order), and with
# `surface_flux = flux_hllc_primitive` and `face2flux = cons2cons`, the numerical flux
# takes the reconstructed primitive variables directly.
n_cells = 8
surface_flux = flux_hllc
reconstruction = ReconstructionWENO5Z()
moment_inversion = MomentInversionIdentity()
center2recon = cons2cons
face2flux = cons2cons
solver = MacroCellFV(n_cells = n_cells, surface_flux = surface_flux,
                     reconstruction = reconstruction,
                     moment_inversion = moment_inversion,
                     center2recon = center2recon, face2flux = face2flux,
                     initial_condition_nodes = 5)

coordinates_min = -5.0
coordinates_max = 5.0
initial_refinement_level = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = false)

# Transmissive boundaries: the ghost subcells mirror the interior subcells
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_condition_do_nothing)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.8)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK33();
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
