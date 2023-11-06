using Trixi
using OrdinaryDiffEq
using LinearAlgebra

###############################################################################
# Definition of the 2D scalar "KPP equation"
#
# See: Kurganov, A., Petrova, G., and Popov, B. (2007).
# Adaptive Semidiscrete Central-Upwind Schemes for Nonconvex Hyperbolic Conservation Laws,
# SIAM Journal on Scientific Computing, 29(6), 2381--2401
# DOI: https://doi.org/10.1137/040614189

struct KPPEquation2D <: Trixi.AbstractEquations{2, 1} end

# The KPP flux is F(u) = (sin(u), cos(u))
@inline function Trixi.flux(u, orientation::Integer, ::KPPEquation2D)
    if orientation == 1
        return SVector(sin(u[1]))
    else
        return SVector(cos(u[1]))
    end
end

# Since the KPP problem is a scalar equation, the entropy-conservative flux is uniquely determined
@inline function Trixi.flux_ec(u_ll, u_rr, orientation::Integer, ::KPPEquation2D)
    # The tolerance of 1e-12 is based on experience and somewhat arbitrarily chosen
    if abs(u_ll[1] - u_rr[1]) < 1e-12
        return 0.5 * (flux(u_ll, orientation, KPPEquation2D()) +
                flux(u_rr, orientation, KPPEquation2D()))
    else
        factor = 1.0 / (u_rr[1] - u_ll[1])
        if orientation == 1
            return SVector(factor * (-cos(u_rr[1]) + cos(u_ll[1])))
        else
            return SVector(factor * (sin(u_rr[1]) - sin(u_ll[1])))
        end
    end
end

# Wavespeeds
@inline wavespeed(::KPPEquation2D) = 1.0
@inline Trixi.max_abs_speeds(u, equation::KPPEquation2D) = (wavespeed(equation),
                                                            wavespeed(equation))
@inline Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equation::KPPEquation2D) = wavespeed(equation)
@inline Trixi.max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equation::KPPEquation2D) = wavespeed(equation) *
                                                                                                           norm(normal_direction)

# Compute entropy: we use the square entropy
@inline Trixi.entropy(u::Real, ::KPPEquation2D) = 0.5 * u^2
@inline Trixi.entropy(u, ::KPPEquation2D) = entropy(u[1], equation)

# Convert between conservative, primitive, and entropy variables. The conserved quantity "u" is also
# considered the "primitive variable". Since we use the square entropy, "u" is also the entropy
# variable.
@inline Trixi.cons2prim(u, ::KPPEquation2D) = u
@inline Trixi.cons2entropy(u, ::KPPEquation2D) = u

Trixi.varnames(::Any, ::KPPEquation2D) = ("u",)

# Standard KPP test problem with discontinuous initial condition
function initial_condition_kpp(x, t, ::KPPEquation2D)
    if x[1]^2 + x[2]^2 < 1
        return SVector(0.25 * 14.0 * pi)
    else
        return SVector(0.25 * pi)
    end
end

###############################################################################
# semidiscretization of the KPP problem
equation = KPPEquation2D()
surface_flux = flux_lax_friedrichs
volume_flux = flux_ec
# Can also compare solution obtained without using entropy-conservative flux. This will not converge
# to the correct (entropy) solution!
# volume_flux = flux_central
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equation, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = first)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

###############################################################################
# Set up the tree mesh (initially a Cartesian grid of [-2,2]^2)
coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                periodicity = true,
                n_cells_max = 500_000)

###############################################################################
# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equation, initial_condition_kpp, solver)

###############################################################################
# Set up adaptive mesh refinement
amr_indicator = IndicatorHennemannGassner(semi,
                                          alpha_max = 1.0,
                                          alpha_min = 0.0001,
                                          alpha_smooth = false,
                                          variable = first)

max_refinement_level = 8

amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, shock_indicator,
                                              base_level = 2,
                                              med_level = 0, med_threshold = 0.0003,
                                              max_level = max_refinement_level,
                                              max_threshold = 0.003,
                                              max_threshold_secondary = shock_indicator.alpha_max)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 1,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 200)

alive_callback = AliveCallback(analysis_interval = 200)

save_solution = SaveSolutionCallback(interval = 200,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2cons)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, amr_callback)

###############################################################################
# run the simulation
sol = solve(ode, SSPRK43(); ode_default_options()..., callback = callbacks)

summary_callback() # Print the timer summary
