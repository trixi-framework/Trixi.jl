using Trixi
using OrdinaryDiffEqSSPRK

###############################################################################
# Geometry & boundary conditions

# Mapping to create a "close-up" mesh around the second quadrant of a cylinder,
# implemented by Georgii Oblapenko. If you use this in your own work, please cite:
#
# - G. Oblapenko and A. Tarnovskiy (2024)
#   Reproducibility Repository for the paper:
#   Entropy-stable fluxes for high-order Discontinuous Galerkin simulations of high-enthalpy flows.
#   [DOI: 10.5281/zenodo.13981615](https://doi.org/10.5281/zenodo.13981615)
#   [GitHub](https://github.com/knstmrd/paper_ec_trixi_chem)
#
# as well as the corresponding paper:
# - G. Oblapenko and M. Torrilhon (2025)
#   Entropy-conservative high-order methods for high-enthalpy gas flows.
#   Computers & Fluids, 2025.
#   [DOI: 10.1016/j.compfluid.2025.106640](https://doi.org/10.1016/j.compfluid.2025.106640)
#
# The mapping produces the following geometry & shock (indicated by the asterisks `* `):
# 
# Here, a boundary layer cell sizing is added to allow for a thermal boundary layer to develop along the cylinder wall.
# 
#                  ____x_neg____
#                 |             |
#               |               |
#             |                 |
#            |                * |
#           |               *   y
#          |   Inflow     *     _
#         |    state    *       p
#         x           *         o
#        _           *          s
#       n           *           |
#      e           *            |
#     g         Shock          .
#     |          *           .
#    |          *          .  <- x_pos
#   |          *          .
#  |          *         .  (Cylinder)
#  |_______y_neg_______.
function mapping_cylinder_shock_fitted(xi_, eta_,
                                       cylinder_radius, spline_points;
                                       wall_normal_stretch = 1.0)
    shock_shape = [
        (spline_points[1], 0.0), # Shock position on the stagnation line (`y_neg`, y = 0)
        (spline_points[2], spline_points[2]), # Shock position at -45° angle
        (0.0, spline_points[3]) # Shock position at outflow (`y_pos`, x = x_max)
    ] # 3 points that define the geometry of the mesh which follows the shape of the shock (known a-priori)
    R = [sqrt(shock_shape[i][1]^2 + shock_shape[i][2]^2) for i in 1:3] # 3 radii

    # Construct spline with form R[1] + c2 * eta_01^2 + c3 * eta_01^3,
    # chosen such that derivative w.r.t eta_01 is 0 at eta_01 = 0 such that
    # we have symmetry along the stagnation line (`y_neg`, y = 0).
    #
    # A single cubic spline doesn't fit the shock perfectly,
    # but is the simplest curve that does a reasonable job and it also can be easily computed analytically.
    # The choice of points on the stagnation line and outflow region is somewhat self-evident
    # (capture the minimum and maximum extent of the shock stand-off),
    # and the point at the 45 degree angle seemed the most logical to add
    # since it only requires one additional value (and not two),
    # simplifies the math a bit, and the angle lies exactly in between the other angles.
    spline_matrix = [1.0 1.0; 0.25 0.125]
    spline_RHS = [R[3] - R[1], R[2] - R[1]]
    spline_coeffs = spline_matrix \ spline_RHS # c2, c3

    eta_01 = (eta_ + 1) / 2 # Transform `eta_` in [-1, 1] to `eta_01` in [0, 1]
    # "Flip" `xi_` in [-1, 1] to `xi_01` in [0, 1] since
    # shock positions where originally for first quadrant, here we use second quadrant
    xi_01 = (-xi_ + 1) / 2

    # Cluster points close to the cylinder wall to mimic a wall-function style
    # boundary-layer spacing while keeping the outer mesh unchanged.
    xi_bl = ifelse(wall_normal_stretch == 1.0,
                   xi_01,
                   (exp(wall_normal_stretch * xi_01) - 1) / (exp(wall_normal_stretch) - 1))

    R_outer = R[1] + spline_coeffs[1] * eta_01^2 + spline_coeffs[2] * eta_01^3

    angle = -π / 4 + eta_ * π / 4 # Angle runs from -90° to 0°
    r = (cylinder_radius + xi_bl * (R_outer - cylinder_radius))

    return SVector(round(r * sin(angle); digits = 8), round(r * cos(angle); digits = 8))
end

# Freestream conditions at 40 km altitude

rho_inf() = 0.00385101 # [kg/m^3]
p_inf() = 277.522 # [Pa]
T_inf() = 250.35 # [K]
mu() = 1.6e-5 # [Pa * s] NOTE: approximate value, temperature dependence is ignored here

@inline function Trixi.temperature(u, equations::CompressibleNavierStokesDiffusion2D)
    rho, rho_v1, rho_v2, rho_e_total = u
    @unpack gamma = equations

    p = (gamma - 1) * (rho_e_total - 0.5f0 * (rho_v1^2 + rho_v2^2) / rho)
    R_specific = 287.058 # [J/(kg*K)]
    T = p / (rho * R_specific) # [K]
    return T
end

# ideal gas employed here, although high-temperature effects become relevant
# see "examples/p4est_2d_dgsem/elixir_euler_therm_perf_cylinder_bowshock_mach6.jl"
gamma() = 1.4
a_inf() = sqrt(gamma() * p_inf() / rho_inf())

M() = 6.0 # [1]
U_inf() = M() * a_inf() # [m/s]

@inline function initial_condition_mach6_flow(x, t, equations)
    return prim2cons(SVector(rho_inf(), U_inf(), 0.0, p_inf()), equations)
end

@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t,
                                                      surface_flux_function,
                                                      equations)
    u_boundary = initial_condition_mach6_flow(x, t, equations)
    return flux(u_boundary, normal_direction, equations)
end

###############################################################################
# Equations, mesh and solver

equations = CompressibleEulerEquations2D(gamma())

prandtl_number() = 0.72
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 1.0,
                                            alpha_min = 0.01,
                                            alpha_smooth = true,
                                            variable = density_pressure)

volume_integral_default = VolumeIntegralWeakForm()
volume_integral_blend_high_order = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_blend_low_order = VolumeIntegralPureLGLFiniteVolume(surface_flux)

volume_integral = VolumeIntegralShockCapturingHGType(shock_indicator;
                                                     volume_integral_default = volume_integral_default,
                                                     volume_integral_blend_high_order = volume_integral_blend_high_order,
                                                     volume_integral_blend_low_order = volume_integral_blend_low_order)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

trees_per_dimension = (60, 30)

cylinder_radius = 0.5
# Follow from a-priori known shock shape, originally for first qaudrant,
# here transformed to second quadrant, see `mapping_cylinder_shock_fitted`.
spline_points = 0.6 .* [1.32, 1.05, 2.25]
cylinder_mapping = (xi, eta) -> mapping_cylinder_shock_fitted(xi, eta,
                                                              cylinder_radius,
                                                              spline_points;
                                                              wall_normal_stretch = 4.0)

mesh = P4estMesh(trees_per_dimension,
                 polydeg = polydeg,
                 mapping = cylinder_mapping,
                 periodicity = false)

# For physical significance of boundary conditions, see sketch at `mapping_cylinder_shock_fitted`
boundary_conditions = (; x_neg = boundary_condition_supersonic_inflow, # Supersonic inflow
                       y_neg = boundary_condition_slip_wall, # Induce symmetry by slip wall
                       y_pos = boundary_condition_do_nothing, # Free outflow
                       x_pos = boundary_condition_slip_wall) # Cylinder

initial_condition = initial_condition_mach6_flow
bc_inflow = BoundaryConditionDirichlet(initial_condition)

# The "Slip" boundary condition rotates all velocities into tangential direction
# and thus acts as a symmetry plane.
ad_heat_bc = Adiabatic((x, t, equations) -> 0.0)
bc_symmetry_plane = BoundaryConditionNavierStokesWall(Slip(), ad_heat_bc)

velocity_bc_noslip = NoSlip((x, t, equations) -> SVector(0.0, 0.0))

rad_eq_heat_bc = RadiativeEquilibriumOneWay(emissivity = 0.85,
                                            T_far_field = T_inf())

boundary_condition_cylinder = BoundaryConditionNavierStokesWall(velocity_bc_noslip,
                                                                rad_eq_heat_bc)

boundary_conditions_parabolic = (; x_neg = bc_inflow,
                                 y_neg = bc_symmetry_plane, # Induce symmetry by slip wall
                                 y_pos = boundary_condition_do_nothing, # Free outflow
                                 x_pos = boundary_condition_cylinder) # Cylinder

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# Semidiscretization & callbacks

tspan = (0.0, 2e-2) # [s]
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 5000)
alive_callback = AliveCallback(alive_interval = 200)

extra_node_variables = (:temperature,)
function Trixi.get_node_variable(::Val{:temperature}, u, mesh, equations, dg, cache,
                                 equations_parabolic, cache_parabolic)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    temp_array = zeros(eltype(cache.elements),
                       n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                       n_elements)

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            temp_array[i, j, element] = Trixi.temperature(u_node, equations_parabolic)
        end
    end

    return temp_array
end

save_solution = SaveSolutionCallback(interval = 5000,
                                     extra_node_variables = extra_node_variables)

amr_controller = ControllerThreeLevel(semi, shock_indicator;
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.175,
                                      max_level = 2, max_threshold = 0.35)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 50,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, amr_callback)

###############################################################################
# Run the simulation

sol = solve(ode, SSPRK43(; thread = Trixi.Threaded());
            dt = 1e-8, abstol = 1e-4, reltol = 1e-4,
            ode_default_options()..., callback = callbacks);
