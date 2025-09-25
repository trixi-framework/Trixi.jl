using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations

prandtl_number() = 0.72
mu = 1e-3 # equivalent to Re = 1000

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number())

# Flow in y-direction to test the symmetry BCs at the left and right boundaries
function initial_condition_freestream(x, t, equations)
    rho = 1.4
    v1 = 0.0
    v2 = 1.0
    p = 1.0

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_freestream

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_hlle)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (false, true))

boundary_conditions = Dict(:x_neg => boundary_condition_slip_wall,
                           :x_pos => boundary_condition_slip_wall)

# The "Slip" boundary condition rotates all velocities into tangential direction
# and thus acts as a symmetry plane.
velocity_bc = Slip()
heat_bc = Adiabatic((x, t, equations_parabolic) -> zero(eltype(x)))
boundary_condition_y = BoundaryConditionNavierStokesWall(velocity_bc,
                                                         heat_bc)

boundary_conditions_parabolic = Dict(:x_neg => boundary_condition_y,
                                     :x_pos => boundary_condition_y)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
