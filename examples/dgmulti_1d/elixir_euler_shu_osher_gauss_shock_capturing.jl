using Trixi
using OrdinaryDiffEq

gamma_gas = 1.4
equations = CompressibleEulerEquations1D(gamma_gas)

###############################################################################
# setup the GSBP DG discretization that uses the Gauss operators from 
# Chan, Del Rey Fernandez, Carpenter (2019). 
# [https://doi.org/10.1137/18M1209234](https://doi.org/10.1137/18M1209234)

# Shu-Osher initial condition for 1D compressible Euler equations
# Example 8 from Shu, Osher (1989).
# [https://doi.org/10.1016/0021-9991(89)90222-2](https://doi.org/10.1016/0021-9991(89)90222-2)
function initial_condition_shu_osher(x, t, equations::CompressibleEulerEquations1D)
    x0 = -4

    rho_left = 27 / 7
    v_left = 4 * sqrt(35) / 9
    p_left = 31 / 3

    # Replaced v_right = 0 to v_right = 0.1 to avoid positivity issues.
    v_right = 0.1
    p_right = 1.0

    rho = ifelse(x[1] > x0, 1 + 1 / 5 * sin(5 * x[1]), rho_left)
    v = ifelse(x[1] > x0, v_right, v_left)
    p = ifelse(x[1] > x0, p_right, p_left)

    return prim2cons(SVector(rho, v, p),
                     equations)
end

initial_condition = initial_condition_shu_osher

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

polydeg = 3
basis = DGMultiBasis(Line(), polydeg, approximation_type = GaussSBP())

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

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; :entire_boundary => boundary_condition)

###############################################################################
#  setup the 1D mesh

cells_per_dimension = (64,)
mesh = DGMultiMesh(dg, cells_per_dimension,
                   coordinates_min = (-5.0,), coordinates_max = (5.0,),
                   periodicity = false)

###############################################################################
#  setup the semidiscretization and ODE problem

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition,
                                    dg, boundary_conditions = boundary_conditions)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

###############################################################################
#  setup the callbacks

# prints a summary of the simulation setup and resets the timers
summary_callback = SummaryCallback()

# analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))

# handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.1)

# collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

# ###############################################################################
# # run the simulation

sol = solve(ode, SSPRK43(), adaptive = true, callback = callbacks, save_everystep = false)
