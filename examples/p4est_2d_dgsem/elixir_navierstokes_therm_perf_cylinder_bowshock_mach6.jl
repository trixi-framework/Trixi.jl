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
                                       cylinder_radius, spline_points)
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

    R_outer = R[1] + spline_coeffs[1] * eta_01^2 + spline_coeffs[2] * eta_01^3

    angle = -π / 4 + eta_ * π / 4 # Angle runs from -90° to 0°
    r = (cylinder_radius + xi_01 * (R_outer - cylinder_radius))

    return SVector(round(r * sin(angle); digits = 8), round(r * cos(angle); digits = 8))
end

# Freestream conditions at 40 km altitude

rho_inf() = 0.00385101 # [kg/m^3]
p_inf() = 277.522 # [Pa]

gamma() = 1.4 # ideal gas employed here, although high-temperature effects become relevant
a_inf() = sqrt(gamma() * p_inf() / rho_inf())

M() = 6.0 # [1]
U_inf() = M() * a_inf() # [m/s]

@inline function initial_condition_mach6_flow(x, t,
                                              equations::NonIdealCompressibleEulerEquations2D)
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

prandtl_number() = 0.72

Re() = 1e6
diameter() = 1.0

t_c = diameter() / U_inf()

prandtl_number() = 0.72
mu() = rho_inf() * U_inf() * diameter() / Re()

equations = CompressibleEulerEquations2D(gamma())
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 1.0,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

trees_per_dimension = (20, 16)

cylinder_radius = 0.5
# Follow from a-priori known shock shape, originally for first qaudrant,
# here transformed to second quadrant, see `mapping_cylinder_shock_fitted`.
spline_points = 0.6 .* [1.32, 1.05, 2.25]
cylinder_mapping = (xi, eta) -> mapping_cylinder_shock_fitted(xi, eta,
                                                              cylinder_radius,
                                                              spline_points)

mesh = P4estMesh(trees_per_dimension,
                 polydeg = polydeg,
                 mapping = cylinder_mapping,
                 periodicity = false)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# For physical significance of boundary conditions, see sketch at `mapping_cylinder_shock_fitted`
boundary_conditions = (; x_neg = boundary_condition_supersonic_inflow, # Supersonic inflow
                       y_neg = boundary_condition_slip_wall, # Induce symmetry by slip wall
                       y_pos = boundary_condition_do_nothing, # Free outflow
                       x_pos = boundary_condition_slip_wall) # Cylinder

initial_condition = initial_condition_mach6_flow
bc_inflow = BoundaryConditionDirichlet(initial_condition)

# The "Slip" boundary condition rotates all velocities into tangential direction
# and thus acts as a symmetry plane.
# TODO: Implement no slip BC that takes velocity from the domain
bc_symmetry_plane = BoundaryConditionNavierStokesWall(Slip(), heat_bc)

velocity_bc_free = NoSlip((x, t, equations) -> SVector(v_in, 0))
# Use adiabatic also on the boundaries to "copy" temperature from the domain
heat_bc_free = Adiabatic((x, t, equations) -> 0)
boundary_condition_free = BoundaryConditionNavierStokesWall(velocity_bc_free, heat_bc_free)

boundary_conditions_parabolic = (; x_neg = bc_inflow,
                       y_neg = bc_symmetry_plane, # Induce symmetry by slip wall
                       y_pos = boundary_condition_do_nothing, # Free outflow
                       x_pos = boundary_condition_slip_wall) # Cylinder

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# Semidiscretization & callbacks

tspan = (0.0, 1e-3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 5000)
alive_callback = AliveCallback(alive_interval = 200)

save_solution = SaveSolutionCallback(interval = 5000)

amr_controller = ControllerThreeLevel(semi, shock_indicator;
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.175,
                                      max_level = 2, max_threshold = 0.35)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 25,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, amr_callback)

###############################################################################
# Run the simulation

sol = solve(ode, SSPRK43(; thread = Trixi.Threaded());
            dt = 1e-7, abstol = 1e-4, reltol = 1e-4,
            ode_default_options()..., callback = callbacks);
