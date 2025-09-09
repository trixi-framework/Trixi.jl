using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
## Semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# Variant of the 4-quadrant Riemann problem considered in 
# - Carsten W. Schulz-Rinne:
#   Classification of the Riemann Problem for Two-Dimensional Gas Dynamics
#   https://doi.org/10.1137/0524006
# and 
# - Carsten W. Schulz-Rinne, James P. Collins, and Harland M. Glaz
#   Numerical Solution of the Riemann Problem for Two-Dimensional Gas Dynamics
#   https://doi.org/10.1137/0914082
function initial_condition_rp(x_, t, equations::CompressibleEulerEquations2D)
    x, y = x_[1], x_[2]

    if x >= 0.5 && y >= 0.5 # 1st quadrant
        rho, v1, v2, p = (0.5313, 0.0, 0.0, 0.4)
    elseif x < 0.5 && y >= 0.5 # 2nd quadrant
        rho, v1, v2, p = (1.0, 0.7276, 0.0, 1.0)
    elseif x < 0.5 && y < 0.5 # 3rd quadrant
        rho, v1, v2, p = (0.8, 0.0, 0.0, 1.0)
    elseif x >= 0.5 && y < 0.5 # 4th quadrant
        rho, v1, v2, p = (1.0, 0.0, 0.7276, 1.0)
    end

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end
initial_condition = initial_condition_rp

function initial_condition_element_discontinuous(x_, t,
                                                 equations::CompressibleEulerEquations2D,
                                                 orientation, direction)
    x, y = x_[1], x_[2]

    if x >= 0.5 && y >= 0.5 # 1st quadrant
        rho, v1, v2, p = (0.5313, 0.0, 0.0, 0.4)
    elseif x < 0.5 && y >= 0.5 # 2nd quadrant
        rho, v1, v2, p = (1.0, 0.7276, 0.0, 1.0)
    elseif x < 0.5 && y < 0.5 # 3rd quadrant
        rho, v1, v2, p = (0.8, 0.0, 0.0, 1.0)
    elseif x >= 0.5 && y < 0.5 # 4th quadrant
        rho, v1, v2, p = (1.0, 0.0, 0.7276, 1.0)
    end

    prim = SVector(rho, v1, v2, p)
    return prim2cons(prim, equations)
end

# See Section 2.3 of the reference below for a discussion of robust
# subsonic inflow/outflow boundary conditions.
#
# - Jan-Reneé Carlson (2011)
#   Inflow/Outflow Boundary Conditions with Application to FUN3D.
#   [NASA TM 20110022658](https://ntrs.nasa.gov/citations/20110022658)
@inline function boundary_condition_subsonic(u_inner, orientation::Integer,
                                             direction, x, t,
                                             surface_flux_function,
                                             equations::CompressibleEulerEquations2D)
    rho_loc, v1_loc, v2_loc, p_loc = cons2prim(u_inner, equations)

    println(typeof(direction))

    p_loc = pressure(initial_condition_rp(x, t, equations), equations)

    prim = SVector(rho_loc, v1_loc, v2_loc, p_loc)
    u_surface = prim2cons(prim, equations)

    return Trixi.flux(u_surface, orientation, equations)
end

# The flow is subsonic at all boundaries.
# Due to the LGL nodes including the boundary points of the reference element ± 1
# setting discontinuous initial conditions with the desired behaviour
#           { u_1, if x <= x_jump
# u(x, t) = {
#           { u_2, if x > x_jump
# is difficult.
# Since the initial condition is queried in `boundary_condition_subsonic` above
# these difficulties propagate to the boundary conditions.
# The setup below is a workaround and works for this specific example, but 
# generalization to other simulations cannot be expected.
boundary_conditions = (x_neg = boundary_condition_subsonic,
                       x_pos = boundary_condition_do_nothing,
                       y_neg = boundary_condition_subsonic,
                       y_pos = boundary_condition_do_nothing)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000,
                periodicity = false)

surface_flux = flux_hllc
volume_flux = flux_ranocha

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
shock_indicator = IndicatorHennemannGassner(equations, basis,
                                            alpha_max = 0.5,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = Trixi.density)

volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral)

# Specialized function for computing coefficients of `func`,
# here the discontinuous `initial_condition_rp`.
#
# Shift the outer (i.e., ±1 on the reference element) nodes passed to the 
# `func` inwards by the smallest amount possible, i.e., [-1 + ϵ, +1 - ϵ].
# This avoids steep gradients in elements if a discontinuity is right at a cell boundary,
# i.e., if the jump location `x_jump` is at the position of an interface which is shared by 
# the nodes x_{e-1}^{(i)} = x_{e}^{(1)}.
#
# In particular, this results in the typically desired behaviour for 
# initial conditions of the form
#           { u_1, if x <= x_jump
# u(x, t) = {
#           { u_2, if x > x_jump
function Trixi.compute_coefficients!(u, func, t,
                                     mesh::TreeMesh{2}, equations, dg::DG, cache)
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            x_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg,
                                           i, j, element)
            if i == 1 # left boundary node
                x_node = SVector(nextfloat(x_node[1]), x_node[2])
            elseif i == nnodes(dg) # right boundary node
                x_node = SVector(prevfloat(x_node[1]), x_node[2])
            end
            if j == 1 # bottom boundary node
                x_node = SVector(x_node[1], nextfloat(x_node[2]))
            elseif j == nnodes(dg) # top boundary node
                x_node = SVector(x_node[1], prevfloat(x_node[2]))
            end

            u_node = func(x_node, t, equations)
            set_node_vars!(u, u_node, equations, dg, i, j, element)
        end
    end
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions = boundary_conditions)

###############################################################################
## ODE solvers, callbacks etc.

tspan = (0.0, 0.25)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.8)

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 3,
                                      med_level = 5, med_threshold = 0.02,
                                      max_level = 8, max_threshold = 0.04)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 10,
                           adapt_initial_condition = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback,
                        stepsize_callback)

###############################################################################
## Run the simulation

sol = solve(ode, SSPRK54();
            dt = 1.0, save_everystep = false, callback = callbacks);
