using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

eos = IdealGas(5 / 3)
eos = PengRobinson()
equations = NonIdealCompressibleEulerEquations2D(eos)

function initial_condition_jet(x, t, equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state

    v1, v2 = 0.0, 0.0

    # rho = 45 # kg/m3
    # T = 290.2 # K, about 4 MPa 

    # from Ma et al
    rho = 45.5
    T = 300

    V = inv(rho)
    e_internal = energy_internal_specific(V, T, eos)
    # cv = 287 / (equations.gamma - 1.0)
    # e_internal = cv * T

    return SVector(rho, rho * v1, rho * v2, rho * (e_internal + 0.5 * (v1^2 + v2^2)))
end

function boundary_condition_jet(x, t, equations::NonIdealCompressibleEulerEquations2D)
    eos = equations.equation_of_state

    h = 0.0022 # 2.2 mm
    if abs(x[2]) < 0.5 * h
        rho = 500 # kg / m3
        v1 = 100 # m/s
        v2 = 0.0
        T = 124.6 # K

        # from Ma et al
        rho = 911
        T = 80

        V = inv(rho)
        e_internal = energy_internal_specific(V, T, eos)

        # # to match pressure inside for ideal gas
        # p_interior = pressure(initial_condition_jet(x, t, equations), equations)
        # e_internal = Trixi.energy_internal(prim2cons(SVector(rho, 0, 0, p_interior), equations),
        #                           equations) / rho

        return SVector(rho, rho * v1, rho * v2, rho * (e_internal + 0.5 * (v1^2 + v2^2)))
    else
        return initial_condition_jet(x, t, equations)
    end
end

# u_jet = boundary_condition_jet([0,0], 0, equations)
# u_interior = initial_condition_jet([0, 0], 0, equations)
# pressure(u_interior, equations)
# pressure(u_jet, equations)

@inline function boundary_condition_subsonic_outflow(u_inner, orientation::Integer,
                                                     direction, x, t,
                                                     surface_flux_function,
                                                     equations::NonIdealCompressibleEulerEquations2D)
    V, v1, v2, T = cons2thermo(u_inner, equations)

    # For subsonic boundary: take pressure from initial condition
    p = pressure(V, T, equations.equation_of_state)

    # invert for new temperature given p, V
    T = 1
    tol = 100 * eps(RealT)
    dp = pressure(V, T, eos) - p
    iter = 1
    while abs(dp) / abs(p) > tol && iter < 100
        dp = pressure(V, T, eos) - p
        dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
        T = max(tol, T - dp / dpdT_V)
        iter += 1
    end
    if iter == 100
        @warn "Solver for temperature(V, p) did not converge"
    end

    thermo = SVector(V, v1, v2, T)
    u_surface = thermo2cons(thermo, equations)

    return flux(u_surface, orientation, equations)
end

initial_condition = initial_condition_jet

volume_flux = flux_central_terashima_etal
surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(3)
indicator = IndicatorEntropyCorrection(equations, basis; scaling = 2.5)
volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolumeO2(basis,
                                                                     volume_flux_fv = surface_flux)

volume_integral = VolumeIntegralEntropyCorrection(volume_integral_default,
                                                  volume_integral_entropy_stable,
                                                  indicator)

solver = DGSEM(basis, surface_flux, volume_integral)

h = 0.0022 # 2.2 mm
coordinates_min = (0.0, -16 * h)
coordinates_max = (32 * h, 16 * h)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 7,
                n_cells_max = 30_000,
                periodicity = (false, true))

boundary_condition_inflow = BoundaryConditionDirichlet(boundary_condition_jet)
# boundary_condition_inflow = boundary_condition_subsonic_inflow

boundary_conditions = (x_neg = boundary_condition_inflow,
                       x_pos = boundary_condition_slip_wall)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1e-3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

###############################################################################
# run the simulation

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)
sol = solve(ode, SSPRK43();
            abstol = 1e-6, reltol = 1e-4,
            saveat = LinRange(tspan..., 10),
            ode_default_options()..., callback = callbacks);

using Plots
plot(ScalarPlotData2D(Trixi.density, sol.u[end], semi), dpi = 300)
