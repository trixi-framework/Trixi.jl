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

# See Section 2.3 of the reference below for a discussion of robust
# subsonic inflow/outflow boundary conditions.
#
# - Jan-Reneé Carlson (2011)
#   Inflow/Outflow Boundary Conditions with Application to FUN3D.
#   [NASA TM 20110022658](https://ntrs.nasa.gov/citations/20110022658)
@inline function boundary_condition_subsonic(u_inner,
                                             normal_direction::AbstractVector,
                                             x, t, surface_flux_function,
                                             equations::CompressibleEulerEquations2D)
    rho_loc, v1_loc, v2_loc, _ = cons2prim(u_inner, equations)
    p_loc = pressure(initial_condition_rp(x, t, equations), equations)
    u_surface = prim2cons(SVector(rho_loc, v1_loc, v2_loc, p_loc), equations)
    return flux(u_surface, normal_direction, equations)
end

# The flow is subsonic at all boundaries.
# For small enough simulation times, the solution remains at the initial condition
# *along the boundaries* of quadrants 2, 3, and 4.
# In quadrants 2 and 4 there are non-zero velocity components (v1 in quadrant 2, v2 in quadrant 4)
# normal to the boundary, which is troublesome for the `boundary_condition_do_nothing`.
# Thus, the `boundary_condition_subsonic` are used instead.
boundary_conditions = (; x_neg = boundary_condition_subsonic,
                       x_pos = boundary_condition_do_nothing,
                       y_neg = boundary_condition_subsonic,
                       y_pos = boundary_condition_do_nothing)

# map [-1,1]^2 to [0,1]^2, with a mild curved warping 
# applied to each quadrant. Mesh interfaces between quadrants 
# are preserved, however. 
function mapping_warp(xi, eta)
    x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return 0.5 * (1 .+ SVector(x, y))
end

trees_per_dimension = (32, 32)
mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 mapping = mapping_warp)

surface_flux = flux_hllc

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator = IndicatorEntropyCorrection(equations, basis)
volume_integral_default = VolumeIntegralWeakForm()
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)
volume_integral = VolumeIntegralAdaptive(indicator,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)

solver = DGSEM(basis, surface_flux, volume_integral)

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
function Trixi.compute_coefficients!(backend::Nothing, u,
                                     func::typeof(initial_condition_rp), t,
                                     mesh::P4estMesh{2}, equations, dg::DG, cache)
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
            Trixi.set_node_vars!(u, u_node, equations, dg, i, j, element)
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

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
## Run the simulation

sol = solve(ode, SSPRK43();
            adaptive = true, abstol = 1e-6, reltol = 1e-4,
            save_everystep = false, callback = callbacks);
