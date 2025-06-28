using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

# Fluid parameters
gamma() = 1.4
prandtl_number() = 0.71

# Parameters for compressible, yet viscous set up
Re() = 100
Ma() = 1.2

# Parameters that can be freely chosen
v_top() = 1
rho_in() = 1
height() = 1.0

# Parameters that follow from Reynolds and Mach number + adiabatic index gamma
nu() = v_top() * height() / Re()

c() = v_top() / Ma()
p_over_rho() = c()^2 / gamma()
p_in() = rho_in() * p_over_rho()
mu() = rho_in() * nu()

equations = CompressibleEulerEquations2D(gamma())
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

# Set inflow, impose known/expected velocity profile
@inline function initial_condition(x, t, equations)
    v1 = x[2] / height() * v_top() # Linear profile from 0 to v_top
    v2 = 0.0

    prim = SVector(rho_in(), v1, v2, p_in())
    return prim2cons(prim, equations)
end

len() = 5 * height() # Roughly constant at this len of the channel
coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (len(), height()) # maximum coordinates (max(x), max(y))

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

velocity_bc_top_left = NoSlip((x, t, equations) -> SVector(x[2] / height() * v_top(), 0))
# Use isothermal for inflow - adiabatic should also work
heat_bc_top_left = Isothermal() do x, t, equations_parabolic
    Trixi.temperature(initial_condition(x, t,
                                        equations_parabolic),
                      equations_parabolic)
end
bc_parabolic_top_left = BoundaryConditionNavierStokesWall(velocity_bc_top_left,
                                                          heat_bc_top_left)

velocity_bc_bottom = NoSlip((x, t, equations) -> SVector(0, 0))
heat_bc_bottom = Adiabatic((x, t, equations) -> 0)
boundary_condition_bottom = BoundaryConditionNavierStokesWall(velocity_bc_bottom,
                                                              heat_bc_bottom)

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

bcs_parabolic = Dict(:x_neg => bc_parabolic_top_left,
                     :x_pos => boundary_condition_copy,
                     :y_neg => boundary_condition_bottom,
                     :y_pos => bc_parabolic_top_left)

solver = DGSEM(polydeg = 3, surface_flux = flux_hll)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver,
                                             boundary_conditions = (bs_hyperbolic,
                                                                    bcs_parabolic))

###############################################################################

tspan = (0, 5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:l2_error_primitive,
                                                              :linf_error_primitive))

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
