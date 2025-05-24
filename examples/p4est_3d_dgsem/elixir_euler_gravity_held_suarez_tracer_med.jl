# Held-Suarez test case
# Following Souza et al.:
# The Flux-Differencing Discontinuous Galerkin Method Applied to an Idealized Fully
# Compressible Nonhydrostatic Dry Atmosphere

using OrdinaryDiffEqLowStorageRK
using Trixi
using LinearAlgebra


function cartesian_to_sphere(x)
    r = norm(x)
    lambda = atan(x[2], x[1])
    phi = asin(x[3] / r)

    return lambda, phi, r
end

function initial_condition_isothermal(x, t, equations::PassiveTracerEquations)
    # equation (60) in the paper
    temperature = 285                     # T_I
    gas_constant = 287                    # R_d
    surface_pressure = 1e5                # p_0
    radius_earth = 6.371229e6             # r_planet
    gravitational_acceleration = 9.80616  # g
    c_v = 717.5                           # specific heat capacity at constant volume

    r = norm(x)
    # Make sure that r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)
    r = z + radius_earth

    # pressure
    # geopotential formulation?
    p = surface_pressure *
        exp(gravitational_acceleration *
            (radius_earth^2 / r - radius_earth) /
            (gas_constant * temperature))

    # density (via ideal gas law)
    rho = p / (gas_constant * temperature)

    # geopotential
    phi = gravitational_acceleration * (radius_earth - radius_earth^2 / r)

    E = c_v * temperature + phi

    tracer = SVector(ntuple(@inline(v -> 0.0001), Val(Trixi.ntracers(equations))))

    return vcat(SVector(rho, 0, 0, 0, rho * E, phi),
                rho*tracer)
end

@inline function source_terms_coriolis(u, x, t,
                                       equations::CompressibleEulerEquationsWithGravity3D)
    radius_earth = 6.371229e6             # r_planet
    angular_velocity = 7.29212e-5         # Ω
    #                  7.27220521664e-05  (Giraldo)

    r = norm(x)
    # Make sure that r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)
    r = z + radius_earth

    du0 = zero(eltype(u))

    # Coriolis term, -2Ω × ρv = -2 * angular_velocity * (0, 0, 1) × u[2:4]
    du2 =  2 * angular_velocity * u[3]
    du3 = -2 * angular_velocity * u[2]

    return SVector(du0, du2, du3, du0, du0, du0)
end

@inline function source_terms_hs_relaxation(u, x, t,
                                            equations::CompressibleEulerEquationsWithGravity3D)
    # equations (55)-(58) in the paper
    secondsperday = 60*60*24
    radius_earth = 6.371229e6   # r_planet
    k_f = 1/secondsperday       # Damping scale for momentum
    k_a = 1/(40*secondsperday)  # Polar relaxation scale
    k_s = 1/(4*secondsperday)   # Equatorial relaxation scale
    T_min = 200                 # Minimum equilibrium temperature
    T_equator = 315             # Equatorial equilibrium temperature
    surface_pressure = 1e5      # p_0
    deltaT = 60                 # Latitudinal temperature difference
    deltaTheta = 10             # Vertical temperature difference
    sigma_b = 0.7               # Dimensionless damping height
    gas_constant = 287          # R_d
    c_v = 717.5                 # Specific heat capacity of dry air at constant volume
    c_p = 1004.5                # Specific heat capacity of dry air at constant pressure
 
    _, _, _, _, pressure = cons2prim(u, equations)
    lon, lat, r = cartesian_to_sphere(x)
    temperature = pressure / (u[1] * gas_constant)

    sigma = pressure / surface_pressure   # "p_0 instead of instantaneous surface pressure"
    delta_sigma = max(0, (sigma-sigma_b)/(1-sigma_b))   # "height factor"
    k_v = k_f * delta_sigma
    k_T = k_a + (k_s - k_a) * delta_sigma * cos(lat)^4

    T_equi = max(T_min,
                 (T_equator - deltaT * sin(lat)^2 - deltaTheta * log(sigma) * cos(lat)^2) *
                  sigma^(gas_constant/c_p))

    # project onto r, normalize! @. Yₜ.c.uₕ -= k_v * Y.c.uₕ
    # Make sure that r is not smaller than radius_earth
    z = max(r - radius_earth, 0.0)
    r = z + radius_earth
    dotprod = (u[2] * x[1] + u[3] * x[2] + u[4] * x[3]) / (r*r)
    
    du2 = -k_v * (u[2] - dotprod * x[1])
    du3 = -k_v * (u[3] - dotprod * x[2])
    du4 = -k_v * (u[4] - dotprod * x[3])

    du5 = -k_T * u[1] * c_v * (temperature - T_equi)

    du0 = zero(eltype(u))
 
    return SVector(du0, du2, du3, du4, du5, du0)
end

# setup
struct RadonDecaySetup{NPRODUCTS}
    dt::Float64
    lambda::Vector{Float64}
    pr1::Array{Float64, 2}
    pr2::Array{Float64, 3}

    function RadonDecaySetup(; nproducts=1, dt=1.0)
        secondsperday = 60*60*24
        half_lifes = [  3.8 * secondsperday,         # 222Rn -> 218Po
                        3.0 * 60,                    # 218Po -> 214Pb
                       27.0 * 60,                    # 214Pb -> 214Bi
                       20.0 * 60,                    # 214Bi -> 214Po -> 210Pb
                       22.0 * 365 * secondsperday ]  # 210Pb -> ... -> 206Pb

        lambda = log(2) ./ half_lifes[1:nproducts]

        pr1 = fill(1.0, nproducts, nproducts)
        for i in 1:nproducts
            for m in 1:i-1
                for q in m:i-1
                    pr1[i,m] = pr1[i,m] * lambda[q]
                end
            end
        end
                   
        pr2 = fill(1.0, nproducts, nproducts, nproducts)
        for i in 1:nproducts
            for m in 1:i-1
                for k in m:i
                    for j in m:i
                        # TODO if (j==k) CYCLE
                        pr2[i,m,k] = pr2[i,m,k] * (lambda[j]-lambda[k])
                    end
                end
            end
        end
        new{nproducts}(dt, lambda, pr1, pr2)
    end
end


@inline function source_terms_radon_decay(u, tracer_equations::PassiveTracerEquations,
                                          setup::RadonDecaySetup{5})
    @unpack dt, lambda, pr1, pr2 = setup
    tracer = tracer_variables(u, tracer_equations)

    tracer_new = zeros(eltype(u), 5)
    for i in 1:5
        tracer_new[i] = tracer[i] * exp( -lambda[i] * dt)
        for m in 1:i-1
            tmp = 0.0
            for k in m:i
                tmp = tmp + exp( -lambda[k] * dt ) / pr2[i,m,k]
            end
            tracer_new[i] = tracer_new[i] + tracer[m] * pr1[i,m] * tmp
        end
    end

    return SVector( (tracer_new .- tracer) ./ dt )
end

@inline function source_terms_radon_decay(u, tracer_equations::PassiveTracerEquations,
                                          setup::RadonDecaySetup{1})
    @unpack dt, lambda = setup
    tau = inv(lambda[1])
    tracer = Trixi.tracers(u, tracer_equations)
    return tracer .* (( exp(-dt/tau ) - 1 ) / dt)
end

@inline function source_terms_radon_init(u, x, setup::RadonDecaySetup{1})
    radius_earth = 6.371229e6   # r_planet
    lon, lat, r = cartesian_to_sphere(x)
    lon_pos = 0.889036791765
    lon_dist = min(abs(lon - lon_pos), abs(lon + 2*pi - lon_pos), abs(lon - 2*pi - lon_pos))
    tracer1 = 1e-6 * exp(-500 * (lon_dist/pi)^2
		         -400 * ((lat - 0.889036791765)/pi)^2 
		         -100 * ((r - radius_earth)/30_000)^2)
    tracer2 = 5e-7 * exp(-300 * (lon_dist/pi)^2
                         -600 * ((lat - 0.4)/pi)^2
                         -100 * ((r - radius_earth)/30_000)^2)
    return SVector(u[1]*tracer1, u[1]*tracer2) 
end

# source terms
@inline function (setup::RadonDecaySetup)(u, x, t,
                                          tracer_equations::PassiveTracerEquations)
    @unpack flow_equations = tracer_equations

    u_flow = Trixi.flow_variables(u, tracer_equations)
    source_terms_flow = source_terms_coriolis(u_flow,x,t,flow_equations) +
                        source_terms_hs_relaxation(u_flow,x,t,flow_equations)
    source_terms_tracer = source_terms_radon_init(u, x, setup) +
                          source_terms_radon_decay(u, tracer_equations, setup)
    return vcat(source_terms_flow, source_terms_tracer)
end

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
flow_equations = CompressibleEulerEquationsWithGravity3D(gamma)

#nproducts = 5
nproducts = 1
equations = PassiveTracerEquations(flow_equations; n_tracers = 2)
radon_setup = RadonDecaySetup(;nproducts=nproducts, dt=1.0) # rough estimate for timestep

initial_condition = initial_condition_isothermal

boundary_conditions = Dict(:inside => Trixi.boundary_condition_slip_wall_noncons,
                           :outside => Trixi.boundary_condition_slip_wall_noncons)

volume_flux = (FluxTracerEquationsCentral(flux_kennedy_gruber),
               Trixi.FluxTracerEquationsPass(flux_nonconservative_waruszewski))
surface_flux = (FluxTracerEquationsCentral(FluxLMARS(340.0)),
#surface_flux = (flux_lax_friedrichs,
                Trixi.FluxTracerEquationsPass(flux_nonconservative_waruszewski))

# Giraldo: (10,8), polydeg 4
lat_lon_trees_per_dim = 16
layers = 8
polydeg = 5

basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 0.002,
                                         alpha_min = 0.0001,
                                         alpha_smooth = true,
                                         variable = density_pressure)
#volume_integral = VolumeIntegralFluxDifferencing(volume_flux)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

mesh = Trixi.P4estMeshCubedSphere(lat_lon_trees_per_dim, layers, 6.371229e6, 30000.0,
                                  polydeg = polydeg, initial_refinement_level = 0)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = radon_setup,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 500 * 24 * 60 * 60) # time in seconds for 1 year
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10000
#analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

output = "out_heldsuarez_tracer_med_HG_cfl08"
save_solution = SaveSolutionCallback(interval = 50000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory = output)

save_restart = SaveRestartCallback(interval = 1000000,
                                   save_final_restart = true,
                                   output_directory = output)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
 #                       analysis_callback,
                        stepsize_callback,
                        alive_callback,
                        save_solution, save_restart)

###############################################################################
# run the simulation

# Use a Runge-Kutta method with automatic (error based) time step size control
# Enable threading of the RK method for better performance on multiple threads
#sol = solve(ode,
#            #RDPK3SpFSAL49(); abstol = 1.0e-8, reltol = 1.0e-8,
#            CarpenterKennedy2N54(williamson_condition = false), dt = 1.0, 
#            ode_default_options()..., maxiters=1e8,
#            callback = callbacks);
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks, maxiters=1e8);
