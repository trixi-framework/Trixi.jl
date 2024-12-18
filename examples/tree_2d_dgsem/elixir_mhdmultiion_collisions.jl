
using OrdinaryDiffEq
using Trixi

###############################################################################
# This elixir describes the frictional slowing of an ionized carbon fluid (C6+) with respect to another species 
# of a background ionized carbon fluid with an initially nonzero relative velocity. It is the second slow-down
# test (fluids with different densities) described in:
# - Ghosh, D., Chapman, T. D., Berger, R. L., Dimits, A., & Banks, J. W. (2019). A 
#   multispecies, multifluid model for laser–induced counterstreaming plasma simulations. 
#   Computers & Fluids, 186, 38-57.
#
# This is effectively a zero-dimensional case because the spatial gradients are zero, and we use it to test the
# collision source terms.
#
# To run this physically relevant test, we use the following characteristic quantities to non-dimensionalize
# the equations:
# Characteristic length: L_inf = 1.00E-03 m (domain size)
# Characteristic density: rho_inf = 1.99E+00 kg/m^3 (corresponds to a number density of 1e20 cm^{-3})
# Characteristic vacuum permeability: mu0_inf = 1.26E-06 N/A^2 (for equations with mu0 = 1)
# Characteristic gas constant: R_inf = 6.92237E+02 J/kg/K (specific gas constant for a Carbon fluid)
# Characteristic velocity: V_inf = 1.00E+06 m/s
#
# The results of the paper can be reproduced using `source_terms = source_terms_collision_ion_ion` (i.e., only
# taking into account ion-ion collisions). However, we include ion-electron collisions assuming a constant 
# electron temperature of 1 keV in this elixir to test the function `source_terms_collision_ion_electron`

# Return the electron pressure for a constant electron temperature Te = 1 keV
function electron_pressure_constantTe(u, equations::IdealGlmMhdMultiIonEquations2D)
    @unpack charge_to_mass = equations
    Te = 0.008029953773 # [nondimensional] = 1 [keV]
    total_electron_charge = zero(eltype(u))
    for k in eachcomponent(equations)
        rho_k = u[3 + (k - 1) * 5 + 1]
        total_electron_charge += rho_k * charge_to_mass[k]
    end

    # Boltzmann constant divided by elementary charge
    kB_e = 7.86319034E-02 #[nondimensional]

    return total_electron_charge * kB_e * Te
end

# Return the constant electron temperature Te = 1 keV
function electron_temperature_constantTe(u, equations::IdealGlmMhdMultiIonEquations2D)
    return 0.008029953773 # [nondimensional] = 1 [keV]
end

# semidiscretization of the ideal MHD equations
equations = IdealGlmMhdMultiIonEquations2D(gammas = (5 / 3, 5 / 3),
                                           charge_to_mass = (76.3049060157692000,
                                                             76.3049060157692000), # [nondimensional]
                                           gas_constants = (1.0, 1.0), # [nondimensional]
                                           molar_masses = (1.0, 1.0), # [nondimensional]
                                           ion_ion_collision_constants = [0.0 0.4079382480442680;
                                                                          0.4079382480442680 0.0], # [nondimensional] (computed with eq (4.142) of Schunk&Nagy (2009))
                                           ion_electron_collision_constants = (8.56368379833E-06,
                                                                               8.56368379833E-06), # [nondimensional] (computed with eq (9) of Ghosh et al. (2019))
                                           electron_pressure = electron_pressure_constantTe,
                                           electron_temperature = electron_temperature_constantTe,
                                           initial_c_h = 0.0) # Deactivate GLM divergence cleaning

# Frictional slowing of an ionized carbon fluid with respect to another background carbon fluid in motion
function initial_condition_slow_down(x, t, equations::IdealGlmMhdMultiIonEquations2D)
    v11 = 0.65508770000000
    v21 = 0.0
    v2 = v3 = 0.0
    B1 = B2 = B3 = 0.0
    rho1 = 0.1
    rho2 = 1.0

    p1 = 0.00040170535986
    p2 = 0.00401705359856

    return prim2cons(SVector(B1, B2, B3, rho1, v11, v2, v3, p1, rho2, v21, v2, v3, p2, 0.0),
                     equations)
end

# Temperature of ion 1
function temperature1(cons, equations::IdealGlmMhdMultiIonEquations2D)
    prim = cons2prim(cons, equations)
    rho, _, _, _, p = Trixi.get_component(1, prim, equations)

    return p / rho / equations.gas_constants[1]
end

# Temperature of ion 2
function temperature2(cons, equations::IdealGlmMhdMultiIonEquations2D)
    prim = cons2prim(cons, equations)
    rho, _, _, _, p = Trixi.get_component(2, prim, equations)

    return p / rho / equations.gas_constants[2]
end

initial_condition = initial_condition_slow_down
tspan = (0.0, 0.1) # 100 [ps]

# Entropy conservative volume numerical fluxes with standard LLF dissipation at interfaces
volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)

solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 1,
                n_cells_max = 1_000_000)

# Ion-ion and ion-electron collision source terms
# In this particular case, we can omit source_terms_lorentz because the magnetic field is zero!
function source_terms(u, x, t, equations::IdealGlmMhdMultiIonEquations2D)
    source_terms_collision_ion_ion(u, x, t, equations) +
    source_terms_collision_ion_electron(u, x, t, equations)
end

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1
analysis_callback = AnalysisCallback(semi,
                                     save_analysis = true,
                                     interval = analysis_interval,
                                     extra_analysis_integrals = (temperature1,
                                                                 temperature2))
alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.01) # Very small CFL due to the stiff source terms 

save_restart = SaveRestartCallback(interval = 100,
                                   save_final_restart = true)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart,
                        stepsize_callback)

###############################################################################

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33();
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary
