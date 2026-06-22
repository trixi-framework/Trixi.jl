using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(5 / 3)

"""
    initial_condition_leblanc_shock_tube(x, t, equations::CompressibleEulerEquations1D)

Leblanc shock tube test case from Section 8.1.1 of
- Lin, Chan, Tomas (2023)
  A positivity preserving strategy for entropy stable discontinuous Galerkin
  discretizations of the compressible Euler and Navier-Stokes equations
  [DOI: 10.1016/j.jcp.2022.111850](https://doi.org/10.1016/j.jcp.2022.111850)
"""
function initial_condition_leblanc_shock_tube(x, t,
                                              equations::CompressibleEulerEquations1D)
    RealT = eltype(x)
    gamma = equations.gamma
    x_interface = RealT(0.33)

    rho_ll = one(RealT)
    v1_ll = zero(RealT)
    p_ll = (gamma - 1) * RealT(1.0e-1)

    rho_rr = RealT(1.0e-3)
    v1_rr = zero(RealT)
    p_rr = (gamma - 1) * RealT(1.0e-10)

    if x[1] < x_interface
        rho = rho_ll
        v1 = v1_ll
        p = p_ll
    elseif x[1] ≈ x_interface
        rho = (rho_ll + rho_rr) / 2
        v1 = (v1_ll + v1_rr) / 2
        p = (p_ll + p_rr) / 2
    else
        rho = rho_rr
        v1 = v1_rr
        p = p_rr
    end

    return prim2cons(SVector(rho, v1, p), equations)
end

initial_condition = initial_condition_leblanc_shock_tube
coordinates_min, coordinates_max = 0.0, 1.0
tspan = (0.0, 2 / 3)

surface_flux = flux_lax_friedrichs
basis = LobattoLegendreBasis(7)
volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha)

solver = DGSEM(basis, surface_flux, volume_integral)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 9,
                periodicity = false)

boundary_conditions = (; x_neg = BoundaryConditionDirichlet(initial_condition),
                       x_pos = boundary_condition_do_nothing)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 5000)
alive_callback = AliveCallback(analysis_interval = 1000)
stepsize_callback = StepsizeCallback(cfl = 0.7)

thresholds = (1.0e-12, 1.0e-12)
variables = (Trixi.density, energy_internal)
local_limiter! = PositivityPreservingLimiterZhangShu(; thresholds, variables)
global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                                      record_davis_yin_iterations = true)

###############################################################################
# run the simulation

callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback,
                        stepsize_callback)

sol = solve(ode,
            RDPK3SpFSAL35(; stage_limiter! = global_limiter!,
                          step_limiter! = global_limiter!);
            adaptive = false, dt = 1.0, # overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
