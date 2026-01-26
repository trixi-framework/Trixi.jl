using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: ForwardDiff

###############################################################################
# semidiscretization of the compressible Euler equations

eos = PengRobinson()
equations = NonIdealCompressibleEulerEquations1D(eos)

@inline function drho_e_dp_at_const_rho(V, T, eos::Trixi.AbstractEquationOfState)
    rho = inv(V)
    dpdT_V, _dpdV_T = Trixi.calc_pressure_derivatives(V, T, eos)
    c_v = Trixi.heat_capacity_constant_volume(V, T, eos)

    # (∂(ρe)/∂p)|ρ = ρ c_v / (∂p/∂T)|V
    return (rho * c_v) / dpdT_V
end

@inline function flux_epec(u_ll, u_rr, orientation::Int,
                           equations::NonIdealCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V_ll, v1_ll, T_ll = cons2prim(u_ll, equations)
    V_rr, v1_rr, T_rr = cons2prim(u_rr, equations)

    rho_ll = u_ll[1]
    rho_rr = u_rr[1]
    rho_e_ll = Trixi.internal_energy_density(u_ll, equations)
    rho_e_rr = Trixi.internal_energy_density(u_rr, equations)
    p_ll = pressure(V_ll, T_ll, eos)
    p_rr = pressure(V_rr, T_rr, eos)

    rho_avg = 0.5f0 * (rho_ll + rho_rr)
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    p_avg = 0.5f0 * (p_ll + p_rr)
    rho_e_avg = 0.5f0 * (rho_e_ll + rho_e_rr)
    p_v1_avg = 0.5f0 * (p_ll * v1_rr + p_rr * v1_ll)

    # chain rule from Terashima
    drho_e_drho_p_ll = Trixi.drho_e_drho_at_const_p(V_ll, T_ll, eos)
    drho_e_drho_p_rr = Trixi.drho_e_drho_at_const_p(V_rr, T_rr, eos)
    drho_e_drho_p_avg = 0.5f0 * (drho_e_drho_p_ll + drho_e_drho_p_rr)
    drho_e_drho_p_rho_avg = 0.5f0 * (drho_e_drho_p_ll * rho_ll + drho_e_drho_p_rr * rho_rr)

    # @variables aL bL aR bR
    # drho_e_drho_p_avg * rho_avg - drho_e_drho_p_rho_avg
    # {a}{b} - {ab} = (aL + aR)(bL + bR) - (aL*bL + aR * bR)
    # (aL*bL + aL * bR + aR * bL + a
    
    rho_e_jump = rho_e_rr - rho_e_ll
    rho_jump = rho_rr - rho_ll
    p_jump = p_rr - p_ll

    drho_e_dp_at_const_rho_ll = drho_e_dp_at_const_rho(V_ll, T_ll, eos)
    drho_e_dp_at_const_rho_rr = drho_e_dp_at_const_rho(V_rr, T_rr, eos)
    drho_e_dp_at_const_rho_avg = 0.5f0 * (drho_e_dp_at_const_rho_ll +
                                          drho_e_dp_at_const_rho_rr)
    num = (rho_e_jump - drho_e_drho_p_avg * rho_jump - drho_e_dp_at_const_rho_avg * p_jump)
    den = drho_e_drho_p_rr - drho_e_drho_p_ll
    rho_avg = rho_avg - num * den / (den^2 + eps(typeof(den)))
    rho_e_avg = (rho_e_avg + drho_e_drho_p_avg * rho_avg - drho_e_drho_p_rho_avg)

    # Ignore orientation since it is always "1" in 1D
    f_rho = rho_avg * v1_avg
    f_rho_v1 = rho_avg * v1_avg * v1_avg + p_avg
    f_rho_E = rho_e_avg * v1_avg + rho_avg * 0.5f0 * (v1_ll * v1_rr) * v1_avg + p_v1_avg

    return SVector(f_rho, f_rho_v1, f_rho_E)
end

initial_condition = Trixi.initial_condition_transcritical_wave

volume_flux = flux_epec
volume_flux = flux_terashima_etal
solver = DGSEM(polydeg = 3, volume_integral = VolumeIntegralFluxDifferencing(volume_flux), 
                surface_flux = flux_lax_friedrichs)

coordinates_min = -0.5
coordinates_max = 0.5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5e-4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
