using OrdinaryDiffEqSSPRK
using Trixi

equations = CompressibleEulerEquations1D(1.4)

"""
    initial_condition_modified_sod(x, t, equations::CompressibleEulerEquations1D)

Modiﬁed Sod shock tube problem, presented in Section 6.4 of Toro's book.
This problem consists of a left sonic rarefaction wave and is useful for testing whether numerical solutions
violate the entropy condition.
An entropy-satisfying solution should produce a smooth(!) rarefaction wave.

## References
- Toro (2009).
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, 3rd Edition.
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

- Lin, Chan (2014)
  High order entropy stable discontinuous Galerkin spectral element methods through subcell limiting
  [DOI: 10.1016/j.jcp.2023.112677](https://doi.org/10.1016/j.jcp.2023.112677)
"""
function initial_condition_modified_sod(x, t, ::CompressibleEulerEquations1D)
    if x[1] < 0.3
        return prim2cons(SVector(1, 0.75, 1), equations)
    else
        return prim2cons(SVector(0.125, 0.0, 0.1), equations)
    end
end
initial_condition = initial_condition_modified_sod

# Using the weak form volume integral gives a wrong solution at the rarefaction wave!
volume_integral_wf = VolumeIntegralWeakForm()

# The entropy-conservative flux-differencing volume integral recovers the rarefaction wave correctly!
volume_integral_fd = VolumeIntegralFluxDifferencing(flux_ranocha)

indicator = IndicatorEntropyChange(maximum_entropy_increase = 0.0)

# Adaptive volume integral using the entropy change indicator to perform the 
# stabilized/EC volume integral when needed and keeping the weak form if it is more diffusive.
volume_integral = VolumeIntegralAdaptive(volume_integral_default = volume_integral_wf,
                                         volume_integral_stabilized = volume_integral_fd,
                                         indicator = indicator)

solver = DGSEM(polydeg = 3, surface_flux = flux_hllc,
               volume_integral = volume_integral)

coordinates_min = 0.0
coordinates_max = 1.0

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 30_000,
                periodicity = false)

# Dirichlet boundary condition is only valid for considered time interval.
# If the rarefaction wave reaches the boundary, this condition is no longer valid!
boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = boundary_condition_do_nothing)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

# For weak form run: Need to enforce positivity explicitly! Not required for flux differencing volume integral.
#=
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

ode_alg = SSPRK43(stage_limiter! = stage_limiter!)
=#
# Flux-differencing volume integral does not require positivity preservation for this test case.
ode_alg = SSPRK43()

sol = solve(ode, ode_alg;
            dt = 4e-4, adaptive = true,
            ode_default_options()..., callback = callbacks);

using Plots

pd = PlotData1D(sol)

# FD
plot(pd["rho"],
     guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
     titlefont = font("Computer Modern", 18), legendfont = font("Computer Modern", 16),
     labelfont = font("Computer Modern", 14),
     linewidth = 2, color = RGB(161/256, 16/256, 53/256),
     label = "Flux-Diff.",
     #title = "Shock Formation Burgers'",
     title = "",
     legend = :bottomleft,
     #yticks = [-20, -10, 0, 10, 20], ylim = (-21, 21),
     xlim = (0, 1.02),
     ylabel = "\$\\rho\$",
     dpi = 600)

# FD + Adaptive
plot!(pd["rho"],
     guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
     titlefont = font("Computer Modern", 18), legendfont = font("Computer Modern", 16),
     labelfont = font("Computer Modern", 14),
     linewidth = 2, color = RGB(0, 84/256, 159/256),
     label = "Adaptive",
     #title = "Shock Formation Burgers'",
     title = "",
     legend = :bottomleft,
     #yticks = [-20, -10, 0, 10, 20], ylim = (-21, 21),
     xlim = (0, 1.02),
     ylabel = "\$\\rho\$",
     dpi = 600)

# Plot for weak form
plot(pd["rho"],
     guidefont = font("Computer Modern", 16), tickfont = font("Computer Modern", 14),
     titlefont = font("Computer Modern", 18), legendfont = font("Computer Modern", 16),
     labelfont = font("Computer Modern", 14),
     linewidth = 2, color = RGB(246/256, 169/256, 0),
     label = "Weak Form",
     #title = "Shock Formation Burgers'",
     title = "",
     legend = :bottomleft,
     #yticks = [-20, -10, 0, 10, 20], ylim = (-21, 21),
     xlim = (0, 1.02),
     ylabel = "\$\\rho\$",
     dpi = 600)