using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# use a continuous version of the blast wave initial condition to avoid 
# floating point issues when evaluating at the interface between the two regions
function initial_condition_weak_C0_blast_wave(x, t,
                                              equations::CompressibleEulerEquations2D)
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0, 0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    rho_outer = one(RealT)
    v1_outer = zero(RealT)
    v2_outer = zero(RealT)
    p_outer = one(RealT)
    rho_inner = 1.1691
    v1_inner = 0.1882 * cos_phi
    v2_inner = 0.1882 * sin_phi
    p_inner = 1.245

    # Calculate primitive variables
    if r > 0.5f0
        rho = rho_outer
        v1 = v1_outer
        v2 = v2_outer
        p = p_outer
    elseif isapprox(r, 0.5f0)
        rho = 0.5f0 * (rho_outer + rho_inner)
        v1 = 0.5f0 * (v1_outer + v1_inner)
        v2 = 0.5f0 * (v2_outer + v2_inner)
        p = 0.5f0 * (p_outer + p_inner)
    else
        rho = rho_inner
        v1 = v1_inner
        v2 = v2_inner
        p = p_inner
    end

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_weak_C0_blast_wave

surface_flux = FluxLaxFriedrichs()
volume_flux = flux_ranocha

polydeg = 3
basis = DGMultiBasis(Quad(), polydeg, approximation_type = GaussSBP())

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

cells_per_dimension = (8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = boundary_condition_periodic)

tspan = (0.0, 0.15)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-6, reltol = 1.0e-6,
            ode_default_options()..., callback = callbacks);
