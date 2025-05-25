using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

# Fluid parameters
const gamma = 1.4
const prandtl_number = 0.72

# Parameters for compressible, yet viscous set up
const Re = 1000
const Ma = 0.8

# Parameters that can be freely chosen
const v_in = 1
const rho_in = 1
const height = 1.0

# Parameters that follow from Reynolds and Mach number + adiabatic index gamma
const nu = v_in * height / Re

const c = v_in / Ma
const p_over_rho = c^2 / gamma
const p_in = rho_in * p_over_rho
const mu = rho_in * nu

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu,
                                                          Prandtl = prandtl_number)

# Set naive inflow
@inline function initial_condition(x, t, equations)
    rho = rho_in
    v1 = v_in
    v2 = 0.0
    p = p_in

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

const len = 10 * height # Roughly constant at this len of the channel
coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (len, height) # maximum coordinates (max(x), max(y))

trees_per_dimension = (36, 12)
mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = (false, false))

# Free outflow
function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                    surface_flux_function,
                                    equations::CompressibleEulerEquations2D)
    # Calculate the boundary flux entirely from the internal solution state
    return Trixi.flux(u_inner, normal_direction, equations)
end

### Hyperbolic boundary conditions ###
bs_hyperbolic = Dict(:x_neg => BoundaryConditionDirichlet(initial_condition), # Weakly enforced inflow BC
                     :x_pos => boundary_condition_outflow, # Free outflow/extended domain
                     # Top/Bottom of channel: Walls
                     :y_neg => boundary_condition_slip_wall,
                     :y_pos => boundary_condition_slip_wall)

### Parabolic boundary conditions ###

velocity_bc_inflow = NoSlip((x, t, equations) -> SVector(v_in, 0))
# Use isothermal for inflow - adiabatic should also work
heat_bc_inflow = Isothermal() do x, t, equations_parabolic
    Trixi.temperature(initial_condition(x, t,
                                        equations_parabolic),
                      equations_parabolic)
end
bc_parabolic_inflow = BoundaryConditionNavierStokesWall(velocity_bc_inflow, heat_bc_inflow)

velocity_bc_wall = NoSlip((x, t, equations) -> SVector(0, 0))
heat_bc_wall = Adiabatic((x, t, equations) -> 0)
boundary_condition_wall = BoundaryConditionNavierStokesWall(velocity_bc_wall, heat_bc_wall)

# On right end: Just copy the state/gradients
@inline function boundary_condition_copy(flux_inner,
                                         u_inner,
                                         normal::AbstractVector,
                                         x, t,
                                         operator_type::Trixi.Gradient,
                                         equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
    return u_inner
end
@inline function boundary_condition_copy(flux_inner,
                                         u_inner,
                                         normal::AbstractVector,
                                         x, t,
                                         operator_type::Trixi.Divergence,
                                         equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
    return flux_inner
end

bcs_parabolic = Dict(:x_neg => bc_parabolic_inflow,
                     :x_pos => boundary_condition_copy,
                     # Top/Bottom of channel: Walls
                     :y_neg => boundary_condition_wall,
                     :y_pos => boundary_condition_wall)

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             boundary_conditions = (bs_hyperbolic,
                                                                    bcs_parabolic))

###############################################################################

tspan = (0, 10)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 1.0,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution)

###############################################################################

time_int_tol = 1e-7
sol = solve(ode, RDPK3SpFSAL49();
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
