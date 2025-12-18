using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.73
mu() = 5e-3
mu() = 1e-6

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())

function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    slope = 15
    amplitude = 0.02
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
                                                          
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(0.5, 0.5)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables
    rho = r > 0.25f0 ? one(RealT) : RealT(1.1691)
    v1 = r > 0.25f0 ? zero(RealT) : RealT(0.1882) * cos_phi
    v2 = r > 0.25f0 ? zero(RealT) : RealT(0.1882) * sin_phi
    # p = r > 0.25f0 ? RealT(1.0E-2) : RealT(1.245)
    p = r > 0.25f0 ? RealT(1.0E-1) : RealT(1.245)

    return prim2cons(SVector(rho, v1, v2, p), equations)
end

function initial_condition_daru(x, t, equations)
    RealT = eltype(x)
    v1 = zero(RealT)
    v2 = zero(RealT)
    rho = x[1] > 0.5f0 ? 1.2 : 120.0
    p = x[1] > 0.5f0 ? 1.2 / equations.gamma : 120.0 / equations.gamma
    # rho = x[1] > 0.5f0 ? 1.2 : 24.0
    # p = x[1] > 0.5f0 ? 1.2 / equations.gamma : 24.0 / equations.gamma

    # rho = 59.4 * tanh(-25*(x[1] - 0.5)) + 60.6
    if abs(x[1] - 0.5f0) < 1e3 * eps()
        rho = 0.5 * (1.2 + 120)
    end
    p = rho / equations.gamma


    return prim2cons(SVector(rho, v1, v2, p), equations)
end

coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))
tspan = (0.0, 1.0)
# initial_condition = initial_condition_blast_wave
# initial_condition = initial_condition_daru

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))
tspan = (0.0, 5.0)
initial_condition = initial_condition_kelvin_helmholtz_instability

dg = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed),
           volume_integral = VolumeIntegralWeakForm())
        #    volume_integral = VolumeIntegralFluxDifferencing(flux_shima_etal))

surface_flux = FluxLaxFriedrichs(max_abs_speed)
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.25,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = flux_shima_etal,
                                                 # volume_flux_dg = flux_central,
                                                 volume_flux_fv = surface_flux)           
dg = DGSEM(basis, surface_flux, volume_integral)


# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                periodicity = (true, true), 
                # periodicity = (false, false), 
                n_cells_max = 200_000) 

# BC types
boundary_condition_noslip_wall = 
    BoundaryConditionNavierStokesWall(NoSlip((x, t, equations_parabolic) -> (0.0, 0.0)), 
                                      Adiabatic((x, t, equations_parabolic) -> 0.0))

# define inviscid boundary conditions
boundary_conditions_hyperbolic = (; x_neg = boundary_condition_slip_wall,
                                    x_pos = boundary_condition_slip_wall,
                                    y_neg = boundary_condition_slip_wall,
                                    y_pos = boundary_condition_slip_wall)

# define viscous boundary conditions
boundary_conditions_parabolic = (; x_neg = boundary_condition_noslip_wall,
                                   x_pos = boundary_condition_noslip_wall,
                                   y_neg = boundary_condition_noslip_wall,
                                   y_pos = boundary_condition_noslip_wall)

if all(mesh.tree.periodicity .== true)
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition, dg) 
    # semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
    #                                              initial_condition, dg)
                                                 
else
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                initial_condition, dg;
                                                boundary_conditions = (boundary_conditions_hyperbolic, 
                                                                        boundary_conditions_parabolic))

    # semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
    #                                             initial_condition, dg;
    #                                             boundary_conditions = (boundary_conditions_hyperbolic, 
    #                                                                     boundary_conditions_parabolic))
end


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
ode = semidiscretize(semi, tspan)

# amr_controller = ControllerThreeLevel(semi, IndicatorLöhner(semi, variable = v1),
#                                       base_level = 3,
#                                       med_level = 5, med_threshold = 0.2,
#                                       max_level = 7, max_threshold = 0.5)
# amr_callback = AMRCallback(semi, amr_controller,
#                            interval = 50,
#                            adapt_initial_condition = true,
#                            adapt_initial_condition_only_refine = true)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback) #, amr_callback)

###############################################################################
# run the simulation

# stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-6, 1.0e-6),
#                                                      variables = (Trixi.density, pressure))
# solver = SSPRK43(stage_limiter!, stage_limiter!)
solver = SSPRK43()

sol = solve(ode, solver; abstol = 1e-6, reltol = 1e-4, 
            ode_default_options()..., callback = callbacks)

using Plots
plot(PlotData2D(sol)["rho"])

u = Trixi.wrap_array(sol.u[end], semi)
T = [Trixi.temperature(get_node_vars(u, equations, dg, i, j, elements), equations_parabolic) 
     for i in eachnode(dg), j in eachnode(dg), elements in eachelement(dg, semi.cache)]
plot(ScalarPlotData2D(T, semi), 
    clims=(0.4, 1.2), xlims=(0.4, 1.0), ylims=(0, 0.25))

# plot!(getmesh(PlotData2D(sol)))