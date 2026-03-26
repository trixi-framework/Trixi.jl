using OrdinaryDiffEqLowStorageRK
using Trixi

polydeg = 2
basis = DGMultiBasis(Tri(), polydeg, approximation_type = SBP())

equations = CompressibleEulerEquations2D(1.4)
@inline function initial_condition_mach2_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = 1.4
    v1 = 2.0
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_mach2_flow

volume_flux = flux_ranocha
surface_flux = flux_hllc
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
dg = DGMulti(basis,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = volume_integral)

# use pre-defined Triangulate geometry in StartUpDG
meshIO = StartUpDG.triangulate_domain(StartUpDG.Scramjet(); h = 0.05)

# the pre-defined Triangulate geometry in StartUpDG has integer boundary tags. this routine
# assigns boundary faces based on these integer boundary tags.
mesh = DGMultiMesh(dg, meshIO, Dict(:wall => 1, :inflow => 2, :outflow => 3))

boundary_conditions = (; inflow = BoundaryConditionDirichlet(initial_condition),
                       outflow = BoundaryConditionDirichlet(initial_condition), # assumes supersonic outflow
                       wall = boundary_condition_slip_wall)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.7 * estimate_dt(mesh, dg),
            ode_default_options()...,
            callback = callbacks);
