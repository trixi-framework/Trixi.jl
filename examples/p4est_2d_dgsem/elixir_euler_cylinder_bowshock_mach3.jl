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
        (spline_points[1], 0.0),
        (spline_points[2], spline_points[2]),
        (0.0, spline_points[3])
    ] # 3 points that define the geometry of the mesh which follows the shape of the shock (known a-priori)
    R = [sqrt(shock_shape[i][1]^2 + shock_shape[i][2]^2) for i in 1:3]  # 3 radii

    # Construct spline with form R[1] + c2 * eta_01^2 + c3 * eta_01^3,
    # chosen such that derivative w.r.t eta_01 is 0 at eta_01 = 0
    spline_matrix = [1.0 1.0; 0.25 0.125]
    spline_RHS = [R[3] - R[1], R[2] - R[1]]
    spline_coeffs = spline_matrix \ spline_RHS # c2, c3

    eta_01 = (eta_ + 1) / 2 # Transform `eta_` in [-1, 1] to `eta_01` in [0, 1]
    xi_01 = (-xi_ + 1) / 2 # "Flip" `xi_` in [-1, 1] to `xi_01` in [0, 1]

    R_outer = R[1] + spline_coeffs[1] * eta_01^2 + spline_coeffs[2] * eta_01^3

    angle = -π / 4 + eta_ * π / 4 # Angle runs from -90° to 0°
    r = (cylinder_radius + xi_01 * (R_outer - cylinder_radius))

    return SVector(round(r * sin(angle); digits = 8), round(r * cos(angle); digits = 8))
end

@inline function initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
    # set the freestream flow parameters
    rho_freestream = equations.gamma
    v1 = 3.0 # => Mach 3 for unity speed of sound
    v2 = 0
    p_freestream = 1
    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t,
                                                      surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach3_flow(x, t, equations)
    flux = Trixi.flux(u_boundary, normal_direction, equations)

    return flux
end

# For physical significance of boundary conditions, see sketch at `mapping_cylinder_shock_fitted`
boundary_conditions = Dict(:x_neg => boundary_condition_supersonic_inflow, # Supersonic inflow
                           :y_neg => boundary_condition_slip_wall, # Induce symmetry by slip wall
                           :y_pos => boundary_condition_do_nothing, # Free outflow 
                           :x_pos => boundary_condition_slip_wall) # Cylinder

###############################################################################
# Equations, mesh and solver

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_hllc
volume_flux = flux_ranocha

shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

trees_per_dimension = (25, 25)

cylinder_radius = 0.5
spline_points = [1.32, 1.05, 2.25] # Follow from a-priori known shock shape
cylinder_mapping = (xi, eta) -> mapping_cylinder_shock_fitted(xi, eta,
                                                              cylinder_radius,
                                                              spline_points)

mesh = P4estMesh(trees_per_dimension,
                 polydeg = polydeg,
                 mapping = cylinder_mapping,
                 periodicity = false)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_mach3_flow,
                                    solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# Semidiscretization & callbacks

tspan = (0.0, 5.0) # More or less stationary shock position reached
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 5000)
alive_callback = AliveCallback(alive_interval = 200)

save_solution = SaveSolutionCallback(dt = 0.25,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi, shock_indicator;
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.175,
                                      max_level = 3, max_threshold = 0.35)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 25,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, amr_callback)

stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-6, 5.0e-6),
                                                     variables = (Trixi.density, pressure))

###############################################################################
# Run the simulation

sol = solve(ode, SSPRK33(stage_limiter! = stage_limiter!, thread = Trixi.True());
            dt = 1.6e-5, # Fixed timestep works decent here
            ode_default_options()..., callback = callbacks);
