using OrdinaryDiffEqSSPRK
using Trixi

surface_flux = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())
volume_flux = flux_ranocha
dg = DGMulti(polydeg = 3, element_type = Line(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

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

cells_per_dimension = (50,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (0.0,), coordinates_max = (1.0,), periodicity = false)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(); adaptive = false,
            dt = 0.5 * estimate_dt(mesh, dg),
            ode_default_options()...,
            callback = callbacks);
