using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

"""
    initial_condition_kelvin_helmholtz_instability_fjordholm_etal(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Ulrik S. Fjordholm, Roger Käppeli, Siddhartha Mishra, Eitan Tadmor (2014)
  Construction of approximate entropy measure valued
  solutions for hyperbolic systems of conservation laws
  [arXiv: 1402.0909](https://arxiv.org/abs/1402.0909)
"""
function initial_condition_kelvin_helmholtz_instability_fjordholm_etal(x, t,
                                                                       equations::CompressibleEulerEquations2D)
    # typical resolution 128^2, 256^2
    # domain size is [0,+1]^2
    # interface is sharp, but randomly perturbed
    # The random numbers used in the initial conditions have been generated as follows:
    #
    # using StableRNGs
    #
    # rng = StableRNG(100)
    #
    # a1 = rand(rng, m)
    # a2 = rand(rng, m)
    # a1 .= a1 / sum(a1)
    # a2 .= a2 / sum(a2)
    # b1 = (rand(rng, m) .- 0.5) .* pi
    # b2 = (rand(rng, m) .- 0.5) .* pi

    m = 10
    a1 = [0.04457096674422902, 0.03891512410182607, 0.0030191053979293433,
        0.0993913172320319,
        0.1622302137588842, 0.1831383653456182, 0.11758003014101702, 0.07964318348142958,
        0.0863245324711805, 0.18518716132585408]
    a2 = [0.061688440856337096, 0.23000237877135882, 0.04453793881833177,
        0.19251530387370916,
        0.11107917357941084, 0.05898041974649702, 0.09949312336096268, 0.07022276346006465,
        0.10670366489014596, 0.02477679264318211]
    b1 = [0.06582340543754152, 0.9857886297001535, 0.8450452205037154, -1.279648120993805,
        0.45454198915209526, -0.13359370986823993, 0.07062615913363897, -1.0097986278512623,
        1.0810669017430343, -0.14207309803877177]
    b2 = [-1.1376882185131414, -1.4798197129947765, 0.6139290513283818, -0.3319087388365522,
        0.14633328999192285, -0.06373231463100072, -0.6270101051216724, 0.13941252226261905,
        -1.0337526453303645, 1.0441408867083155]
    Y1 = 0.0
    Y2 = 0.0
    for n in 1:m
        Y1 += a1[n] * cos(b1[n] + 2 * n * pi * x[1])
        Y2 += a2[n] * cos(b2[n] + 2 * n * pi * x[1])
    end

    J1 = 0.25
    J2 = 0.75
    epsilon = 0.01
    I1 = J1 + epsilon * Y1
    I2 = J2 + epsilon * Y2

    if (x[2] > I1) && (x[2] < I2)
        rho = 2
        v1 = -0.5
    else
        rho = 1
        v1 = 0.5
    end
    v2 = 0
    p = 2.5

    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability_fjordholm_etal

surface_flux = flux_hllc
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.001,
                                         alpha_min = 0.0001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 6,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 400
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 400,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution)

###############################################################################
# run the simulation
sol = solve(ode, SSPRK43();
            ode_default_options()..., callback = callbacks);
summary_callback() # print the timer summary
