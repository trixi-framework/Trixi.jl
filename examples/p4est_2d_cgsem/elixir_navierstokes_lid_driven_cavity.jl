# Lid-driven cavity flow with the continuous Galerkin spectral element method (CGSEM),
# cf. examples/p4est_2d_dgsem/elixir_navierstokes_lid_driven_cavity.jl

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 0.001

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

# Create CGSEM solver with polynomial degree = 3. CGSEM has no interface dissipation, so
# an entropy-conservative flux differencing volume integral is used for robustness.
# The surface flux enters only through the weakly imposed boundary conditions.
solver = CGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined, conforming mesh
trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3, initial_refinement_level = 2,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (false, false))

function initial_condition_cavity(x, t, equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    Ma = convert(RealT, 0.1)
    rho = 1
    u, v = 0, 0
    p = 1 / (Ma^2 * equations.gamma)
    return prim2cons(SVector(rho, u, v, p), equations)
end
initial_condition = initial_condition_cavity

# BC types
velocity_bc_lid = NoSlip((x, t, equations_parabolic) -> SVector(one(eltype(x)),
                                                                zero(eltype(x))))
velocity_bc_cavity = NoSlip((x, t, equations_parabolic) -> SVector(zero(eltype(x)),
                                                                   zero(eltype(x))))
heat_bc = Adiabatic((x, t, equations_parabolic) -> zero(eltype(x)))
boundary_condition_lid = BoundaryConditionNavierStokesWall(velocity_bc_lid, heat_bc)
boundary_condition_cavity = BoundaryConditionNavierStokesWall(velocity_bc_cavity, heat_bc)

# define inviscid boundary conditions
boundary_conditions = (; x_neg = boundary_condition_slip_wall,
                       y_neg = boundary_condition_slip_wall,
                       y_pos = boundary_condition_slip_wall,
                       x_pos = boundary_condition_slip_wall)

# define parabolic boundary conditions
boundary_conditions_parabolic = (; x_neg = boundary_condition_cavity,
                                 y_neg = boundary_condition_cavity,
                                 y_pos = boundary_condition_lid,
                                 x_pos = boundary_condition_cavity)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 25.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(thread = Trixi.Threaded());
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
