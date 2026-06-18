# Runner for Mach 3 cylinder variants on P4estMesh.
#
# Toggle these options directly, or override them with `trixi_include` kwargs:
#   trixi_include(".../elixir_euler_supersonic_cylinder_runner.jl";
#                 use_ecav = true,
#                 use_volume_correction = false,
#                 use_shock_capturing = true,
#                 shock_capturing_alpha_max = 0.5)

using OrdinaryDiffEqSSPRK
using LinearAlgebra: I
using Trixi

use_ecav = true
use_volume_correction = !use_ecav
use_shock_capturing = true;
use_positivity_limiter = false
shock_capturing_alpha_max = 1.0;
refine = 4
Ma = 2.0
num_trial = 1

polydeg = 3
final_time = 10.0
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
    v1 = Ma
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

boundary_conditions_hyperbolic = (; Bottom = boundary_condition_slip_wall,
                                  Circle = boundary_condition_slip_wall,
                                  Top = boundary_condition_slip_wall,
                                  Right = boundary_condition_outflow,
                                  Left = boundary_condition_supersonic_inflow)

###############################################################################
# artificial-viscosity/parabolic setup

prandtl_number() = 0.73
mu() = 0.0

equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())

# The physical parabolic equation is only used to provide gradient variables and
# storage for ECAV in this inviscid example.
boundary_conditions_parabolic = (; Bottom = boundary_condition_do_nothing,
                                 Circle = boundary_condition_do_nothing,
                                 Top = boundary_condition_do_nothing,
                                 Right = boundary_condition_do_nothing,
                                 Left = boundary_condition_do_nothing)

solver_parabolic = ParabolicFormulationLocalDG()

###############################################################################
# mesh

mesh_file = joinpath(@__DIR__, "CylinderSuperSonicMa2.0.inp")

mesh = P4estMesh{2}(mesh_file, initial_refinement_level=refine)

###############################################################################
# DGSEM solver

volume_flux = flux_central
surface_flux = flux_lax_friedrichs

basis = LobattoLegendreBasis(polydeg)
VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
filter = ones(polydeg + 1)

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

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

if use_positivity_limiter
    stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                         variables = (pressure,
                                                                      Trixi.density))
    sol = solve(ode, SSPRK43(; stage_limiter!);
                abstol = abstol, reltol = reltol, saveat = saveat,
                ode_default_options()..., callback = callbacks)
else
    sol = solve(ode, SSPRK43();
                abstol = abstol, reltol = reltol, saveat = saveat,
                ode_default_options()..., callback = callbacks)
end

using Plots
using JLD2

@save "Cylinder" * string(num_trial) * ".jld2" sol semi
data = load("Cylinder1.jld2")

function density_schlieren(u_ode, semi; beta = 10.0)
    (; solver, cache) = semi
    (; derivative_matrix) = solver.basis
    (; contravariant_vectors, inverse_jacobian) = cache.elements
    u = Trixi.wrap_array_native(u_ode, semi)
    schlieren = similar(inverse_jacobian)

    for element in Trixi.eachelement(solver, cache),
        j in Trixi.eachnode(solver), i in Trixi.eachnode(solver)

        drho_dxi = zero(eltype(schlieren))
        drho_deta = zero(eltype(schlieren))

        for ii in Trixi.eachnode(solver)
            drho_dxi += derivative_matrix[i, ii] * u[1, ii, j, element]
            drho_deta += derivative_matrix[j, ii] * u[1, i, ii, element]
        end

        Ja11, Ja12 = Trixi.get_contravariant_vector(1, contravariant_vectors,
                                                    i, j, element)
        Ja21, Ja22 = Trixi.get_contravariant_vector(2, contravariant_vectors,
                                                    i, j, element)
        inv_jacobian = inverse_jacobian[i, j, element]

        drho_dx = inv_jacobian * (Ja11 * drho_dxi + Ja21 * drho_deta)
        drho_dy = inv_jacobian * (Ja12 * drho_dxi + Ja22 * drho_deta)
        schlieren[i, j, element] = sqrt(drho_dx^2 + drho_dy^2)
    end

    schlieren_min, schlieren_max = extrema(schlieren)
    if schlieren_max > schlieren_min
        @. schlieren = exp(-beta * (schlieren - schlieren_min) /
                           (schlieren_max - schlieren_min))
    else
        fill!(schlieren, one(eltype(schlieren)))
    end

    return schlieren
end

gr() # good backend for GIFs

anim = @animate for k in eachindex(sol.u)
    schlieren = density_schlieren(sol.u[k], semi; beta = 10.0)
    pd = ScalarPlotData2D(schlieren, semi; variable_name = "density Schlieren")
    plot(pd;
         title = "density Schlieren, t = $(round(sol.t[k], digits = 3))",
         aspect_ratio = :equal,
         clims = (0.0, 1.0),
         color = :grays)
end

gif(anim, "rho_schlieren.gif", fps = 10)