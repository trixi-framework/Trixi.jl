using OrdinaryDiffEqLowStorageRK
using Trixi
using Plots

##############################################################################################
# initial condition

equations = LinearVariableScalarAdvectionEquation2D()

# initial condition, round density cloud is defined
@inline function initial_condition_schaer_mountain_cloud(x, t, equations)
    RealT = eltype(x)
    x_0, z_0 = -50_000, 9_000
    rho_0 = 1
    A_x, A_z = 25_000, 3000

    r = sqrt(((x[1] - x_0) / A_x)^2 + ((x[2] - z_0) / A_z)^2)

    if r <= 1
        rho = convert(RealT, rho_0 * (cos((pi * r) / 2))^2)
    else
        rho = 0.0f0
    end
    return SVector(rho)
end

# wind profile in horizontal direktion
@inline function velocity_schaer_mountain(x, equations)
    RealT = eltype(x)
    u_0 = 10.0f0
    z_1 = 4000.0f0
    z_2 = 5000.0f0

    if x[2] <= z_1
        u_1 = 0.0f0
    elseif z_2 <= x[2]
        u_1 = 1.0f0
    else
        u_1 = convert(RealT, sin((pi / 2) * (x[2] - z_1) / (z_2 - z_1))^2)
    end
    return SVector(u_1 * u_0, 0.0f0)
end

# mesh transformation
function mapping(xi_, eta_)
    # transformation from [-1,1]x[-1,1] to [-150_000,150_000]x[0,25_000]
    xi = xi_ * 150_000
    eta = eta_ * 12_500 + 12_500

    # upper boundary
    H = 25_000

    # topography
    h_c = 3_000
    lambda_c = 8_000
    a_c = 4_000

    h_star = 0
    if abs(xi) <= a_c
        h_star = h_c * cospi(0.5 * xi / a_c )^2
    end
#    topo = cospi(xi / lambda_c)^2 * h_star
    topo = h_star

    x = xi
    # linear blending to top
    y = topo + (H - topo) * eta / H

    return SVector(x, y)
end

##############################################################################################
# semidiscretization
polydeg = 3

initial_condition = initial_condition_schaer_mountain_cloud

# flux and solver 
surface_flux = flux_lax_friedrichs
volume_integral = VolumeIntegralFluxDifferencing(flux_central)

solver = DGSEM(polydeg = polydeg,
               surface_flux = surface_flux)
               #volume_integral = volume_integral)

hohqmesh = false
if hohqmesh
    # P4est HOHQ mesh
    mesh_file = joinpath(@__DIR__, "schaer_mountain_test.inp")
    mesh = P4estMesh{2}(mesh_file, polydeg = polydeg)

    # boundary conditions
    boundary_conditions_dirichlet = Dict(
        :left => BoundaryConditionDirichlet(initial_condition),
        :right => BoundaryConditionDirichlet(initial_condition),
        :top => BoundaryConditionDirichlet(initial_condition),
        :bottom => BoundaryConditionDirichlet(initial_condition))

    output_name = "out_cloud_advection_hohqmesh"
else
    nodes_1d = polydeg + 1
    cells_per_dimension = div.((300, 50), 2)
    # TODO
    #cells_per_dimension = (150, 50)
    mesh = P4estMesh(cells_per_dimension; polydeg = polydeg, mapping = mapping,
                     periodicity = false)

    # boundary conditions
    boundary_conditions_dirichlet = Dict(
        :x_neg => BoundaryConditionDirichlet(initial_condition),
        :x_pos => BoundaryConditionDirichlet(initial_condition),
        :y_neg => BoundaryConditionDirichlet(initial_condition),
        :y_pos => BoundaryConditionDirichlet(initial_condition))

    output_name = "out_cloud_advection_tranform_llf"
end

# the velocity is passed as auxiliary_field into the cache
semi = SemidiscretizationHyperbolic(mesh,
                                    equations,
                                    initial_condition,
                                    solver,
                                    boundary_conditions = boundary_conditions_dirichlet,
                                    aux_field = velocity_schaer_mountain)

##############################################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10000.0)

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
solution_variables = cons2prim

analysis_callback = AnalysisCallback(semi,
                                     interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     output_directory = output_name,
                                     solution_variables = solution_variables)

stepsize_callback = StepsizeCallback(cfl = 0.25)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 0,
                                      med_level = 1, med_threshold = 0.7,
                                      max_level = 2, max_threshold = 0.9)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 50,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

visualization = VisualizationCallback(semi; interval = 100) #, show_mesh = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        #amr_callback,
                        #visualization,
                        save_solution)

###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 25.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
