using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

surface_flux = FluxPlusDissipation(flux_ranocha, DissipationMatrixWintersEtal())
volume_flux = flux_ranocha
dg = DGMulti(polydeg = 3, element_type = Line(), approximation_type = SBP(),
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

equations = CompressibleEulerEquations1D(1.4)

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
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

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
