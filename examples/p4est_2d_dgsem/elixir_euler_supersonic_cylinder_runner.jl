# Runner for Mach 3 cylinder variants on P4estMesh.
#
# Toggle these options directly, or override them with `trixi_include` kwargs:
#   trixi_include(".../elixir_euler_supersonic_cylinder_runner.jl";
#                 use_ecav = true,
#                 use_volume_correction = false,
#                 use_shock_capturing = true,
#                 shock_capturing_alpha_max = 0.5)
using Revise
using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: I
using Trixi

use_ecav = false;
use_volume_correction = true;
use_shock_capturing = false;
use_positivity_limiter = true;
shock_capturing_alpha_max = 0.1;
Ma = 1.5
refine = 3
num_trial = 1

polydeg = 3
final_time = 5.0
analysis_interval = 1000
save_interval = 1000
abstol = 1.0e-7
reltol = 1.0e-5
saveat = 0.05

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
    rho_freestream = 1.4
    v1 = 1.1
    v2 = 0.0
    p_freestream = 1.0

    prim = SVector(rho_freestream, v1, v2, p_freestream)
    return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach3_flow

@inline function boundary_condition_supersonic_inflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    u_boundary = initial_condition_mach3_flow(x, t, equations)
    return flux(u_boundary, normal_direction, equations)
end

@inline function boundary_condition_outflow(u_inner, normal_direction::AbstractVector, x, t,
                                            surface_flux_function,
                                            equations::CompressibleEulerEquations2D)
    return flux(u_inner, normal_direction, equations)
end

@inline function boundary_condition_vary_outflow(u_inner,
                                                      normal_direction::AbstractVector,
                                                      x, t, surface_flux_function,
                                                      equations::CompressibleEulerEquations2D)
    rho, rho_v1, rho_v2, rho_e_total = u_inner
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v = sqrt(v1 ^2 + v2^2)
    p = (equations.gamma - 1) * (rho_e_total - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
    c  = sqrt(equations.gamma * p / rho)
    if v >= c
        return flux(u_inner, normal_direction, equations)
    else
        #assume non dimensionalized free stream values
        p_freestream = 1.0
        return flux(prim2cons(SVector(rho, v1, v2, p_freestream), equations), normal_direction, equations)
    end

end

boundary_conditions_hyperbolic = (; Bottom = boundary_condition_slip_wall,
                                  Circle = boundary_condition_slip_wall,
                                  Top = boundary_condition_slip_wall,
                                  Right = boundary_condition_vary_outflow,
                                  Left = boundary_condition_supersonic_inflow)

###############################################################################
# artificial-viscosity/parabolic setup
prandtl_number() = 0.73
mu() = 0.0

equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
equations_schlieren = CompressibleNavierStokesDiffusion2D(equations, mu = 0.0,
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesConservative())

boundary_condition_inflow = BoundaryConditionDirichlet(initial_condition)
heat_bc = Adiabatic((x, t, equations_parabolic) -> 0.0)
boundary_condition_parabolic_slip_wall = BoundaryConditionNavierStokesWall(Slip(), heat_bc)
boundary_conditions_parabolic = (; Bottom = boundary_condition_parabolic_slip_wall,
                                 Circle = boundary_condition_parabolic_slip_wall,
                                 Top = boundary_condition_parabolic_slip_wall,
                                 Right = boundary_condition_do_nothing,
                                 Left = boundary_condition_inflow)

solver_parabolic = ParabolicFormulationLocalDG()

###############################################################################
# mesh

mesh_suffix = polydeg == 1 ? "N1" : ""
mesh_file = joinpath(@__DIR__, "CylinderSuperSonicMa" * string(1.5) * mesh_suffix * ".inp")
#mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/a08f78f6b185b63c3baeff911a63f628/raw/addac716ea0541f588b9d2bd3f92f643eb27b88f/abaqus_cylinder_in_channel.inp",
#                           joinpath(@__DIR__, "abaqus_cylinder_in_channel.inp"))

mesh = P4estMesh{2}(mesh_file, initial_refinement_level=refine)

###############################################################################
# DGSEM solver

volume_flux = flux_central
surface_flux = flux_lax_friedrichs

basis = LobattoLegendreBasis(polydeg)
VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
filter = ones(polydeg + 1)
1
if use_volume_correction
    indicator_ec = IndicatorEntropyCorrection(equations, basis)

    if use_shock_capturing
        indicator_sc = IndicatorHennemannGassner(equations, basis;
                                                 alpha_max = shock_capturing_alpha_max,
                                                 alpha_min = 0.001,
                                                 alpha_smooth = true,
                                                 variable = density_pressure)
        indicator = IndicatorEntropyCorrectionShockCapturingCombined(indicator_entropy_correction = indicator_ec,
                                                                     indicator_shock_capturing = indicator_sc)
    else
        indicator = indicator_ec
    end

    volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)
    volume_integral_stabilized = VolumeIntegralPureLGLFiniteVolume(volume_flux_fv = surface_flux)
    volume_integral = VolumeIntegralAdaptive(indicator,
                                             volume_integral_default,
                                             volume_integral_stabilized)
elseif use_shock_capturing
    shock_indicator = IndicatorHennemannGassner(equations, basis;
                                                alpha_max = shock_capturing_alpha_max,
                                                alpha_min = 0.001,
                                                alpha_smooth = true,
                                                variable = density_pressure)
    volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                    volume_flux_dg = volume_flux,
                                                    volume_flux_fv = surface_flux)
else
    volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
end

solver = DGSEM(basis, surface_flux, volume_integral)

if use_ecav
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 VDM = VDM, filter = filter,
                                                 combine_rhs = Trixi.True(),
                                                 solver_parabolic = solver_parabolic,
                                                 boundary_conditions = (boundary_conditions_hyperbolic,
                                                                        boundary_conditions_parabolic))
else
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_conditions_hyperbolic)
end

###############################################################################
# ODE solvers

tspan = (0.0, final_time)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = save_interval,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
#stepsize_callback = StepsizeCallback(cfl = 0.1)
callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

if use_positivity_limiter
    local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                         variables = (pressure,
                                                                      Trixi.density))
    global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)
    @show "positive on"
    sol = solve(ode, SSPRK43(; stage_limiter! = global_limiter!,
                step_limiter! = global_limiter!); #dt = 0.01, adaptive=false,
                abstol = abstol, reltol = reltol, saveat = saveat,
                ode_default_options()..., callback = callbacks)
else
    #sol = solve(ode, RDPK3SpFSAL49();
    #           abstol = abstol, reltol = reltol, saveat = saveat,
    #            ode_default_options()..., callback = callbacks)
    # sol = solve(ode, SSPRK43();
    #             abstol = abstol, reltol = reltol, saveat = saveat,
    #             ode_default_options()..., callback = callbacks)
    sol = solve(ode, SSPRK43(); 
                abstol = abstol, reltol = reltol, saveat = saveat,
                ode_default_options()..., callback = callbacks)
    
end

using Plots
using JLD2

@save "Cylinder" * string(num_trial) * ".jld2" sol semi
data = load("Cylinder1.jld2")

function edge_mach_numbers(u_ode, semi, x_max)
    (; equations, solver, cache) = semi
    (; node_coordinates) = cache.elements
    u = Trixi.wrap_array_native(u_ode, semi)

    tolerance = 1.0e-10 * max(1.0, abs(x_max))
    mach_numbers = Float64[]

    for element in Trixi.eachelement(solver, cache),
        j in Trixi.eachnode(solver), i in Trixi.eachnode(solver)

        x = node_coordinates[1, i, j, element]
        abs(x - x_max) <= tolerance || continue

        rho = u[1, i, j, element]
        rho_v1 = u[2, i, j, element]
        rho_v2 = u[3, i, j, element]
        rho_e_total = u[4, i, j, element]

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        p = (equations.gamma - 1) *
            (rho_e_total - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
        sound_speed = sqrt(equations.gamma * p / rho)

        push!(mach_numbers, sqrt(v1^2 + v2^2) / sound_speed)
    end

    return mach_numbers
end

(; node_coordinates) = semi.cache.elements
x_min = minimum(view(node_coordinates, 1, :, :, :))
x_max = maximum(view(node_coordinates, 1, :, :, :))
for i in 1:length(sol.u)
    right_edge_mach = edge_mach_numbers(sol.u[i], semi, x_max)
    println("Right-edge Mach number at t = $(sol.t[i]): ",
            "min = $(minimum(right_edge_mach)), ",
            "mean = $(sum(right_edge_mach) / length(right_edge_mach)), ",
            "max = $(maximum(right_edge_mach))")
end
for i in 1:length(sol.u)
    right_edge_mach = edge_mach_numbers(sol.u[i], semi, x_min)
    println("Right-edge Mach number at t = $(sol.t[i]): ",
            "min = $(minimum(right_edge_mach)), ",
            "mean = $(sum(right_edge_mach) / length(right_edge_mach)), ",
            "max = $(maximum(right_edge_mach))")
end

function schlieren_normalize!(schlieren; beta = 10.0)
    schlieren_min, schlieren_max = extrema(schlieren)
    if schlieren_max > schlieren_min
        @. schlieren = exp(-beta * (schlieren - schlieren_min) /
                           (schlieren_max - schlieren_min))
    else
        fill!(schlieren, one(eltype(schlieren)))
    end
    return schlieren
end

function setup_schlieren_gradient(semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    schlieren_solver = ParabolicFormulationLocalDG()

    if semi isa SemidiscretizationArtificialViscosity
        return (; equations_schlieren, schlieren_solver,
                cache_parabolic = semi.cache_parabolic,
                boundary_conditions_parabolic = semi.boundary_conditions_parabolic)
    elseif semi isa SemidiscretizationHyperbolicParabolic
        return (; equations_schlieren, schlieren_solver,
                cache_parabolic = semi.cache_parabolic,
                boundary_conditions_parabolic = semi.boundary_conditions_parabolic)
    else
        uEltype = eltype(cache.elements.inverse_jacobian)
        cache_parabolic = Trixi.create_cache_parabolic(mesh, equations, dg,
                                                     nelements(dg, cache), uEltype)
        return (; equations_schlieren, schlieren_solver, cache_parabolic,
                boundary_conditions_parabolic)
    end
end

"""
    density_schlieren_dg(u_ode, semi, schlieren_context; beta=10.0, t=0.0)

Compute density Schlieren using Trixi's Local-DG gradient (`calc_gradient!`) on
curved `P4estMesh`, including volume, interface, and boundary terms.
"""
function density_schlieren_dg(u_ode, semi, schlieren_context; beta = 10.0, t = 0.0)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array_native(u_ode, semi)

    (; equations_schlieren, schlieren_solver, cache_parabolic,
     boundary_conditions_parabolic) = schlieren_context
    (; u_transformed, gradients) = cache_parabolic.parabolic_container
    gradients_x, gradients_y = gradients

    Trixi.transform_variables!(u_transformed, u, mesh, equations_schlieren, dg, cache)
    Trixi.calc_gradient!(gradients, u_transformed, t, mesh, equations_schlieren,
                         boundary_conditions_parabolic, dg, schlieren_solver, cache)

    schlieren = similar(cache.elements.inverse_jacobian)

    for element in Trixi.eachelement(dg, cache),
        j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)

        drho_dx = gradients_x[1, i, j, element]
        drho_dy = gradients_y[1, i, j, element]
        schlieren[i, j, element] = sqrt(drho_dx^2 + drho_dy^2)
    end

    return schlieren_normalize!(schlieren; beta)
end

schlieren_context = setup_schlieren_gradient(semi)

gr() # good backend for GIFs

anim = @animate for k in eachindex(sol.u)
    schlieren = density_schlieren_dg(sol.u[k], semi, schlieren_context;
                                     beta = 10.0, t = sol.t[k])
    pd = ScalarPlotData2D(schlieren, semi; variable_name = "density Schlieren (LDG)")
    plot(pd;
         title = "density Schlieren, t = $(round(sol.t[k], digits = 3))",
         aspect_ratio = :equal,
         clims = (0.0, 1.0),
         color = :grays)
end

gif(anim, "rho_schlieren_dg.gif", fps = 10)

