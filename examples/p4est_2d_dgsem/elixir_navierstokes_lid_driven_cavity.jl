using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 0.001

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh
trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3, initial_refinement_level = 2,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (false, false))

function initial_condition_cavity(x, t, equations::CompressibleEulerEquations2D)
    Ma = 0.1
    rho = 1.0
    u, v = 0.0, 0.0
    p = 1.0 / (Ma^2 * equations.gamma)
    return prim2cons(SVector(rho, u, v, p), equations)
end
initial_condition = initial_condition_cavity

# BC types
velocity_bc_lid = NoSlip((x, t, equations) -> SVector(1.0, 0.0))
velocity_bc_cavity = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_lid = BoundaryConditionNavierStokesWall(velocity_bc_lid, heat_bc)
boundary_condition_cavity = BoundaryConditionNavierStokesWall(velocity_bc_cavity, heat_bc)

boundary_conditions = Dict(:x_neg => boundary_condition_slip_wall,
                           :y_neg => boundary_condition_slip_wall,
                           :y_pos => boundary_condition_slip_wall,
                           :x_pos => boundary_condition_slip_wall)

boundary_conditions_parabolic = Dict(:x_neg => boundary_condition_cavity,
                                     :y_neg => boundary_condition_cavity,
                                     :y_pos => boundary_condition_lid,
                                     :x_pos => boundary_condition_cavity)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 25.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary
