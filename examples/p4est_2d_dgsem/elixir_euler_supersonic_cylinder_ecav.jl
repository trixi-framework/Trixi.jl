# Channel flow around a cylinder at Mach 3 using entropy-correction artificial viscosity.
#
# This variant intentionally disables AMR so that the current P4est ECAV
# implementation only sees curved conforming elements.

using OrdinaryDiffEqSSPRK
using LinearAlgebra: I
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

@inline function initial_condition_mach3_flow(x, t, equations::CompressibleEulerEquations2D)
    rho_freestream = 1.4
    v1 = 1.5
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
# storage for ECAV in this inviscid example, so use neutral parabolic boundaries.
boundary_conditions_parabolic = (; Bottom = boundary_condition_do_nothing,
                                 Circle = boundary_condition_do_nothing,
                                 Top = boundary_condition_do_nothing,
                                 Right = boundary_condition_do_nothing,
                                 Left = boundary_condition_do_nothing)

solver_parabolic = ParabolicFormulationLocalDG()

###############################################################################
# mesh

mesh_file = Trixi.download("https://gist.githubusercontent.com/andrewwinters5000/a08f78f6b185b63c3baeff911a63f628/raw/addac716ea0541f588b9d2bd3f92f643eb27b88f/abaqus_cylinder_in_channel.inp",
                           joinpath(@__DIR__, "abaqus_cylinder_in_channel.inp"))

#mesh = P4estMesh{2}(mesh_file, initial_refinement_level=1)
mesh = P4estMesh{2}(mesh_file)

###############################################################################
# DGSEM solver

volume_flux = flux_central
surface_flux = flux_lax_friedrichs

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
filter = ones(polydeg + 1)

shock_indicator = IndicatorHennemannGassner(equations, basis;
                                            alpha_max = 1.0,
                                            alpha_min = 0.001,
                                            alpha_smooth = true,
                                            variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(shock_indicator;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

indicator_ec = IndicatorEntropyCorrection(equations, basis)

## non shock capturing version
#volume_integral_default = VolumeIntegralWeakForm()
#volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)

## shock capturing version with limiting
indicator_ec = IndicatorEntropyCorrection(equations, basis)
indicator_sc = IndicatorHennemannGassner(equations, basis;
                                         alpha_max = 0.5,
                                         alpha_min = 0.001,
                                         alpha_smooth = true,
                                         variable = density_pressure)

indicator = IndicatorEntropyCorrectionShockCapturingCombined(
    indicator_entropy_correction = indicator_ec,
    indicator_shock_capturing = indicator_sc)

volume_integral_default = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral_entropy_stable = VolumeIntegralPureLGLFiniteVolume(surface_flux)

volume_integral = VolumeIntegralAdaptive(indicator,
                                         volume_integral_default,
                                         volume_integral_entropy_stable)

## for non shock capturing
#volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = flux_lax_friedrichs,
            volume_integral = volume_integral)


semi = SemidiscretizationHyperbolic(mesh, equations,
                                        initial_condition, solver; 
                                        boundary_conditions = boundary_conditions_hyperbolic)

## Artificial viscosity
semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             VDM = VDM, filter = filter,
                                             combine_rhs = Trixi.True(),
                                             solver_parabolic = solver_parabolic,
                                             boundary_conditions = (boundary_conditions_hyperbolic,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)
saveat = range(tspan[1], tspan[2], length = 81)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution)

# Positivity limiter is still necessary for this strong-shock example.
stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (5.0e-7, 1.0e-6),
                                                     variables = (pressure, Trixi.density))

###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(stage_limiter!); abstol=1e-6,
        reltol=1e-4, saveat=saveat, ode_default_options()..., callback = callbacks);

using Plots

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

gr()
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