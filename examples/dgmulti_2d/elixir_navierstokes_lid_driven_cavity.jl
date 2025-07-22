using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 0.001

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the 
# `StepsizeCallback` (CFL-Condition) and less diffusion.
dg = DGMulti(polydeg = 3, element_type = Quad(), approximation_type = GaussSBP(),
             surface_integral = SurfaceIntegralWeakForm(FluxLaxFriedrichs(max_abs_speed_naive)),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

top(x, tol = 50 * eps()) = abs(x[2] - 1) < tol
rest_of_boundary(x, tol = 50 * eps()) = !top(x, tol)
is_on_boundary = Dict(:top => top, :rest_of_boundary => rest_of_boundary)

cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension; is_on_boundary)

function initial_condition_cavity(x, t, equations::CompressibleEulerEquations2D)
    Ma = 0.1
    rho = 1.0
    u, v = 0.0, 0.0
    p = 1.0 / (Ma^2 * equations.gamma)
    return prim2cons(SVector(rho, u, v, p), equations)
end
initial_condition = initial_condition_cavity

# BC types
velocity_bc_lid = NoSlip((x, t, equations_parabolic) -> SVector(1.0, 0.0))
velocity_bc_cavity = NoSlip((x, t, equations_parabolic) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations_parabolic) -> 0.0)
boundary_condition_lid = BoundaryConditionNavierStokesWall(velocity_bc_lid, heat_bc)
boundary_condition_cavity = BoundaryConditionNavierStokesWall(velocity_bc_cavity, heat_bc)

# define inviscid boundary conditions
boundary_conditions = (; :top => boundary_condition_slip_wall,
                       :rest_of_boundary => boundary_condition_slip_wall)

# define viscous boundary conditions
boundary_conditions_parabolic = (; :top => boundary_condition_lid,
                                 :rest_of_boundary => boundary_condition_cavity)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 10.0)
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

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
