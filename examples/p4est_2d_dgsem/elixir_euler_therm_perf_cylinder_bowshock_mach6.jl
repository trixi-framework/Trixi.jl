using Trixi
using OrdinaryDiffEqSSPRK
using ForwardDiff

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

@inline function initial_condition_mach6_flow(x, t,
                                              equations::NonIdealCompressibleEulerEquations2D)
    RealT = eltype(x)
    eos = equations.equation_of_state

    # Freestream conditions at 40 km altitude
    rho = 0.00385101 # [kg/m^3]
    V = inv(rho) # [m^3/kg]

    p = 277.522 # [Pa]

    # invert for temperature given p, V
    T = temperature_given_Vp(V, p, eos; initial_T = 251.05, # [K]
                             tol = 100 * eps(RealT), maxiter = 100)

    a = Trixi.speed_of_sound(V, T, eos)
    M = 6.0 # [1]
    v1 = M * a # [m/s]
    v2 = 0.0 # [m/s]

    return thermo2cons(SVector(V, v1, v2, T), equations)
end

@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t,
                                                      surface_flux_function,
                                                      equations)
    u_boundary = initial_condition_mach6_flow(x, t, equations)
    return flux(u_boundary, normal_direction, equations)
end

# For physical significance of boundary conditions, see sketch at `mapping_cylinder_shock_fitted`
boundary_conditions = (; x_neg = boundary_condition_supersonic_inflow, # Supersonic inflow
                       y_neg = boundary_condition_slip_wall, # Induce symmetry by slip wall
                       y_pos = boundary_condition_do_nothing, # Free outflow
                       x_pos = boundary_condition_slip_wall) # Cylinder

###############################################################################
# Equations, mesh and solver

# Data taken from https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf page 276/284
M = 0.0289651159 # [kg/mol]
R_universal = 8.31446261815324 # [J/(mol K)]
R_specific = R_universal / M # [J/(kg K)]

temp_bounds = SVector(200.0, 1000.0, 6000.0) # [K]

a_cold = [1.009950160e+04; -1.968275610e+02; 5.009155110e+00; -5.761013730e-03;
          1.066859930e-05; -7.940297970e-09; 2.185231910e-12; -1.767967310e+02;
          -3.921504225e+00]
a_hot = [2.415214430e+05; -1.257874600e+03; 5.144558670e+00; -2.138541790e-04;
         7.065227840e-08; -1.071483490e-11; 6.577800150e-16; 6.462263190e+03;
         -8.147411905e+00]
a_ = hcat(a_cold, a_hot)
a = Trixi.SMatrix{9, 2}(a_)

eos = ThermallyPerfectGas9PolyFit(R_specific = R_specific,
                                  temperature_bounds = temp_bounds,
                                  a = a)

equations = NonIdealCompressibleEulerEquations2D(eos)

# Reduce tolerance to speed things up (otherwise, `eos_newton_maxiter` would need to be increased)
Trixi.eos_newton_tol(eos::ThermallyPerfectGas9PolyFit) = 1e-5

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_lax_friedrichs
volume_flux = flux_terashima_etal

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

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_mach6_flow, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
# Semidiscretization & callbacks

tspan = (0.0, 1e-3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 5000)
alive_callback = AliveCallback(alive_interval = 200)

# Add `:gamma` to `extra_node_variables` tuple ...
extra_node_variables = (:gamma,)

# ... and specify the function `get_node_variable` for this symbol,
# with first argument matching the symbol (turned into a type via `Val`) for dispatching.
function Trixi.get_node_variable(::Val{:gamma}, u, mesh, equations, dg, cache)
    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    gamma_array = zeros(eltype(cache.elements),
                        n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                        n_elements)

    eos = equations.equation_of_state

    # We can accelerate the computation by thread-parallelizing the loop over elements
    # by using the `@threaded` macro.
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            # Get temperature
            u_node_thermo = cons2thermo(u_node, equations)
            T = u_node_thermo[4]

            gamma_array[i, j, element] = Trixi.gamma(T, eos)
        end
    end

    return gamma_array
end

save_solution = SaveSolutionCallback(interval = 5000,
                                     solution_variables = cons2thermo,
                                     extra_node_variables = extra_node_variables)

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
