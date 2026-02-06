using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler multicomponent equations

# 1) Dry Air  2) Helium + 28% Air
equations = CompressibleEulerMulticomponentEquations2D(gammas = (1.4, 1.648),
                                                       gas_constants = (0.287 * 10^3, 1.578 * 10^3))

"""
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{5, 2})

A shock-bubble testcase for multicomponent Euler equations
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t,
                                        equations::CompressibleEulerMulticomponentEquations2D{5,
                                                                                              2})
    RealT = eltype(x)
    @unpack gas_constants = equations

    # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
    delta = convert(RealT, 0.03)

    # Region I
    rho1_1 = delta
    rho2_1 = RealT(1.225) * gas_constants[1] / gas_constants[2] - delta
    v1_1 = zero(RealT)
    v2_1 = zero(RealT)
    p_1 = 101325

    # Region II
    rho1_2 = RealT(1.225) - delta
    rho2_2 = delta
    v1_2 = zero(RealT)
    v2_2 = zero(RealT)
    p_2 = 101325

    # Region III
    rho1_3 = RealT(1.6861) - delta
    rho2_3 = delta
    v1_3 = -RealT(113.5243)
    v2_3 = zero(RealT)
    p_3 = 159060

    # Set up Region I & II:
    inicenter = SVector(0.225, 0.0445) # center of bubble
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    if (x[1] > 0.275)
        # Set up Region III
        rho1 = rho1_3
        rho2 = rho2_3
        v1 = v1_3
        v2 = v2_3
        p = p_3
    elseif (r < 0.025f0)
        # Set up Region I
        rho1 = rho1_1
        rho2 = rho2_1
        v1 = v1_1
        v2 = v2_1
        p = p_1
    else
        # Set up Region II
        rho1 = rho1_2
        rho2 = rho2_2
        v1 = v1_2
        v2 = v2_2
        p = p_2
    end

    return prim2cons(SVector(v1, v2, p, rho1, rho2), equations)
end
initial_condition = initial_condition_shock_bubble

surface_flux = flux_lax_friedrichs

volume_flux = flux_ranocha
basis = LobattoLegendreBasis(3) # 5
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral_stabilized = VolumeIntegralShockCapturingRRG(basis, indicator_sc;
                                                             volume_flux_dg = volume_flux,
                                                             volume_flux_fv = surface_flux,
                                                             slope_limiter = vanleer)

volume_integral = VolumeIntegralAdaptive(volume_integral_default = VolumeIntegralWeakForm(),
                                         volume_integral_stabilized = volume_integral_stabilized,
                                         indicator = nothing) # Indicator taken from `volume_integral_stabilized`

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (0.445, 0.089)
trees_per_dimension = (8, 2)
mesh = P4estMesh(trees_per_dimension, polydeg = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 5, periodicity = false)
    
restart_file = "out/restart_000101714.h5"
#mesh = load_mesh(restart_file)

bc_LR = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:x_neg => bc_LR, :x_pos => bc_LR,
                           :y_neg => boundary_condition_slip_wall,
                           :y_pos => boundary_condition_slip_wall,)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# Plot times in paper above
# TODO: Somehow our simulation is wrong, i.e., lagging behind
t1 = 23.32e-6
t1_exp = 32e-6

tspan = (0.0, t1_exp)

ode = semidiscretize(semi, tspan)

t2 = 42.98e-6
t3 = 52.81e-6
t4 = 67.55e-6
t5 = 77.38e-6
t6 = 101.95e-6
t7 = 259.21e-6

tspan = (load_time(restart_file), load_time(restart_file))
ode = semidiscretize(semi, tspan, restart_file)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (Trixi.density,))

alive_callback = AliveCallback(alive_interval = 100)

amr_indicator = IndicatorLöhner(semi, variable = Trixi.density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level = 0,
                                      med_level = 3, med_threshold = 0.001,
                                      max_level = 5, max_threshold = 0.002)

amr_callback = AMRCallback(semi, amr_controller,
                           interval = 100,
                           adapt_initial_condition = false)

extra_node_variables = (:density, :schlieren,)

function Trixi.get_node_variable(::Val{:density}, u, mesh, equations, dg, cache)

    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    density_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    max_abs_gradient = zero(eltype(u))
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
            density_array[i, j, element] = u_node[4] + u_node[5]
        end
    end

    return density_array
end

function Trixi.get_node_variable(::Val{:schlieren}, u, mesh, equations, dg, cache)

    n_nodes = nnodes(dg)
    n_elements = nelements(dg, cache)
    # By definition, the variable must be provided at every node of every element!
    # Otherwise, the `SaveSolutionCallback` will crash.
    schlieren_array = zeros(eltype(cache.elements),
                            n_nodes, n_nodes, # equivalent: `ntuple(_ -> n_nodes, ndims(mesh))...,`
                            n_elements)

    @unpack contravariant_vectors, inverse_jacobian = cache.elements
    @unpack derivative_matrix = dg.basis

    max_abs_gradient = zero(eltype(u))
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            # Get the contravariant vectors Ja^1 and Ja^2
            Ja11, Ja12 = Trixi.get_contravariant_vector(1, contravariant_vectors, i, j, element)
            Ja21, Ja22 = Trixi.get_contravariant_vector(2, contravariant_vectors, i, j, element)

            rho_d_xi = zero(eltype(u))  # ∂ρ/∂ξ
            rho_d_eta = zero(eltype(u)) # ∂ρ/∂η

            for k in eachnode(dg)
                u_kj = Trixi.get_node_vars(u, equations, dg, k, j, element)
                u_ik = Trixi.get_node_vars(u, equations, dg, i, k, element)

                density_kj = u_kj[4] + u_kj[5]
                density_ik = u_ik[4] + u_ik[5]

                rho_d_xi += derivative_matrix[i, k] * density_kj
                rho_d_eta += derivative_matrix[j, k] * density_ik
            end
            # Transform to physical coordinates using contravariant vectors
            # ∇ρ_x = Ja11 * ∂ρ/∂ξ + Ja21 * ∂ρ/∂η
            # ∇ρ_y = Ja12 * ∂ρ/∂ξ + Ja22 * ∂ρ/∂η
            density_gradient_x = Ja11 * rho_d_xi + Ja21 * rho_d_eta
            density_gradient_y = Ja12 * rho_d_xi + Ja22 * rho_d_eta
            # Multiply by Jacobian inverse to get physical gradient
            inv_jac = cache.elements.inverse_jacobian[i, j, element]
            abs_density_gradient = sqrt(density_gradient_x^2 + density_gradient_y^2) * inv_jac

            if abs_density_gradient > max_abs_gradient
                max_abs_gradient = abs_density_gradient
            end

            schlieren_array[i, j, element] = abs_density_gradient
        end
    end

    k = 1 # scaling factor for schlieren visualization, can be adjusted
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            schlieren_array[i, j, element] = exp(- k * schlieren_array[i, j, element] / max_abs_gradient)
        end
    end

    return schlieren_array
end

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = extra_node_variables)

save_restart = SaveRestartCallback(interval = 100_000,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        #save_restart,
                        #amr_callback,
                        save_solution,
                        )

###############################################################################
# run the simulation

sol = solve(ode, SSPRK432(thread = Trixi.True());
            dt = 2e-10, ode_default_options()...,
            abstol = 1e-5, reltol = 1e-5,
            maxiters = Inf, callback = callbacks);
