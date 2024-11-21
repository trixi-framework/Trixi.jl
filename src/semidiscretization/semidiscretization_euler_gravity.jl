# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    ParametersEulerGravity(; background_density=0.0,
                             gravitational_constant=1.0,
                             cfl=1.0,
                             resid_tol=1.0e-4,
                             n_iterations_max=10^4,
                             timestep_gravity=timestep_gravity_erk52_3Sstar!)

Set up parameters for the gravitational part of a [`SemidiscretizationEulerGravity`](@ref).
"""
struct ParametersEulerGravity{RealT <: Real, TimestepGravity}
    background_density     :: RealT # aka rho0
    gravitational_constant :: RealT # aka G
    cfl                    :: RealT
    resid_tol              :: RealT
    n_iterations_max       :: Int
    timestep_gravity       :: TimestepGravity
end

function ParametersEulerGravity(; background_density = 0.0,
                                gravitational_constant = 1.0,
                                cfl = 1.0,
                                resid_tol = 1.0e-4,
                                n_iterations_max = 10^4,
                                timestep_gravity = timestep_gravity_erk52_3Sstar!)
    background_density, gravitational_constant, cfl, resid_tol = promote(background_density,
                                                                         gravitational_constant,
                                                                         cfl, resid_tol)
    ParametersEulerGravity(background_density, gravitational_constant, cfl, resid_tol,
                           n_iterations_max, timestep_gravity)
end

function Base.show(io::IO, parameters::ParametersEulerGravity)
    @nospecialize parameters # reduce precompilation time

    print(io, "ParametersEulerGravity(")
    print(io, "background_density=", parameters.background_density)
    print(io, ", gravitational_constant=", parameters.gravitational_constant)
    print(io, ", cfl=", parameters.cfl)
    print(io, ", n_iterations_max=", parameters.n_iterations_max)
    print(io, ", timestep_gravity=", parameters.timestep_gravity)
    print(io, ")")
end
function Base.show(io::IO, ::MIME"text/plain", parameters::ParametersEulerGravity)
    @nospecialize parameters # reduce precompilation time

    if get(io, :compact, false)
        show(io, parameters)
    else
        setup = [
            "background density (ρ₀)" => parameters.background_density,
            "gravitational constant (G)" => parameters.gravitational_constant,
            "CFL (gravity)" => parameters.cfl,
            "max. #iterations" => parameters.n_iterations_max,
            "time integrator" => parameters.timestep_gravity
        ]
        summary_box(io, "ParametersEulerGravity", setup)
    end
end

"""
    SemidiscretizationEulerGravity

A struct containing everything needed to describe a spatial semidiscretization
of a the compressible Euler equations with self-gravity, reformulating the
Poisson equation for the gravitational potential as steady-state problem of
the hyperblic diffusion equations.
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  "A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics"
  [arXiv: 2008.10593](https://arXiv.org/abs/2008.10593)
"""
struct SemidiscretizationEulerGravity{SemiEuler, SemiGravity,
                                      Parameters <: ParametersEulerGravity, Cache} <:
       AbstractSemidiscretization
    semi_euler          :: SemiEuler
    semi_gravity        :: SemiGravity
    parameters          :: Parameters
    performance_counter :: PerformanceCounter
    gravity_counter     :: PerformanceCounter
    cache               :: Cache

    function SemidiscretizationEulerGravity{SemiEuler, SemiGravity, Parameters, Cache}(semi_euler::SemiEuler,
                                                                                       semi_gravity::SemiGravity,
                                                                                       parameters::Parameters,
                                                                                       cache::Cache) where {
                                                                                                            SemiEuler,
                                                                                                            SemiGravity,
                                                                                                            Parameters <:
                                                                                                            ParametersEulerGravity,
                                                                                                            Cache
                                                                                                            }
        @assert ndims(semi_euler) == ndims(semi_gravity)
        @assert typeof(semi_euler.mesh) == typeof(semi_gravity.mesh)
        @assert polydeg(semi_euler.solver) == polydeg(semi_gravity.solver)

        performance_counter = PerformanceCounter()
        gravity_counter = PerformanceCounter()

        new(semi_euler, semi_gravity, parameters, performance_counter, gravity_counter,
            cache)
    end
end

"""
    SemidiscretizationEulerGravity(semi_euler::SemiEuler, semi_gravity::SemiGravity, parameters)

Construct a semidiscretization of the compressible Euler equations with self-gravity.
`parameters` should be given as [`ParametersEulerGravity`](@ref).
"""
function SemidiscretizationEulerGravity(semi_euler::SemiEuler,
                                        semi_gravity::SemiGravity,
                                        parameters) where
         {Mesh,
          SemiEuler <:
          SemidiscretizationHyperbolic{Mesh, <:AbstractCompressibleEulerEquations},
          SemiGravity <:
          SemidiscretizationHyperbolic{Mesh, <:AbstractHyperbolicDiffusionEquations}}
    u_ode = compute_coefficients(zero(real(semi_gravity)), semi_gravity)
    du_ode = similar(u_ode)
    u_tmp1_ode = similar(u_ode)
    u_tmp2_ode = similar(u_ode)
    cache = (; u_ode, du_ode, u_tmp1_ode, u_tmp2_ode)

    SemidiscretizationEulerGravity{typeof(semi_euler), typeof(semi_gravity),
                                   typeof(parameters), typeof(cache)}(semi_euler,
                                                                      semi_gravity,
                                                                      parameters, cache)
end

function remake(semi::SemidiscretizationEulerGravity;
                uEltype = real(semi.semi_gravity.solver),
                semi_euler = semi.semi_euler,
                semi_gravity = semi.semi_gravity,
                parameters = semi.parameters)
    semi_euler = remake(semi_euler, uEltype = uEltype)
    semi_gravity = remake(semi_gravity, uEltype = uEltype)

    # Recreate cache, i.e., registers for u with e.g. AD datatype
    u_ode = compute_coefficients(zero(real(semi_gravity)), semi_gravity)
    du_ode = similar(u_ode)
    u_tmp1_ode = similar(u_ode)
    u_tmp2_ode = similar(u_ode)
    cache = (; u_ode, du_ode, u_tmp1_ode, u_tmp2_ode)

    SemidiscretizationEulerGravity{typeof(semi_euler), typeof(semi_gravity),
                                   typeof(parameters), typeof(cache)}(semi_euler,
                                                                      semi_gravity,
                                                                      parameters, cache)
end

function Base.show(io::IO, semi::SemidiscretizationEulerGravity)
    @nospecialize semi # reduce precompilation time

    print(io, "SemidiscretizationEulerGravity using")
    print(io, semi.semi_euler)
    print(io, ", ", semi.semi_gravity)
    print(io, ", ", semi.parameters)
    print(io, ", cache(")
    for (idx, key) in enumerate(keys(semi.cache))
        idx > 1 && print(io, " ")
        print(io, key)
    end
    print(io, "))")
end

function Base.show(io::IO, mime::MIME"text/plain", semi::SemidiscretizationEulerGravity)
    @nospecialize semi # reduce precompilation time

    if get(io, :compact, false)
        show(io, semi)
    else
        summary_header(io, "SemidiscretizationEulerGravity")
        summary_line(io, "semidiscretization Euler",
                     semi.semi_euler |> typeof |> nameof)
        show(increment_indent(io), mime, semi.semi_euler)
        summary_line(io, "semidiscretization gravity",
                     semi.semi_gravity |> typeof |> nameof)
        show(increment_indent(io), mime, semi.semi_gravity)
        summary_line(io, "parameters", semi.parameters |> typeof |> nameof)
        show(increment_indent(io), mime, semi.parameters)
        summary_footer(io)
    end
end

# The compressible Euler semidiscretization is considered to be the main semidiscretization.
# The hyperbolic diffusion equations part is only used internally to update the gravitational
# potential during an rhs! evaluation of the flow solver.
@inline function mesh_equations_solver_cache(semi::SemidiscretizationEulerGravity)
    mesh_equations_solver_cache(semi.semi_euler)
end

@inline Base.ndims(semi::SemidiscretizationEulerGravity) = ndims(semi.semi_euler)

@inline Base.real(semi::SemidiscretizationEulerGravity) = real(semi.semi_euler)

# computes the coefficients of the initial condition
@inline function compute_coefficients(t, semi::SemidiscretizationEulerGravity)
    compute_coefficients!(semi.cache.u_ode, t, semi.semi_gravity)
    compute_coefficients(t, semi.semi_euler)
end

# computes the coefficients of the initial condition and stores the Euler part in `u_ode`
@inline function compute_coefficients!(u_ode, t, semi::SemidiscretizationEulerGravity)
    compute_coefficients!(semi.cache.u_ode, t, semi.semi_gravity)
    compute_coefficients!(u_ode, t, semi.semi_euler)
end

@inline function calc_error_norms(func, u, t, analyzer,
                                  semi::SemidiscretizationEulerGravity, cache_analysis)
    calc_error_norms(func, u, t, analyzer, semi.semi_euler, cache_analysis)
end

function rhs!(du_ode, u_ode, semi::SemidiscretizationEulerGravity, t)
    @unpack semi_euler, semi_gravity, cache = semi

    u_euler = wrap_array(u_ode, semi_euler)
    du_euler = wrap_array(du_ode, semi_euler)
    u_gravity = wrap_array(cache.u_ode, semi_gravity)

    time_start = time_ns()

    # standard semidiscretization of the compressible Euler equations
    @trixi_timeit timer() "Euler solver" rhs!(du_ode, u_ode, semi_euler, t)

    # compute gravitational potential and forces
    @trixi_timeit timer() "gravity solver" update_gravity!(semi, u_ode)

    # add gravitational source source_terms to the Euler part
    if ndims(semi_euler) == 1
        @views @. du_euler[2, .., :] -= u_euler[1, .., :] * u_gravity[2, .., :]
        @views @. du_euler[3, .., :] -= u_euler[2, .., :] * u_gravity[2, .., :]
    elseif ndims(semi_euler) == 2
        @views @. du_euler[2, .., :] -= u_euler[1, .., :] * u_gravity[2, .., :]
        @views @. du_euler[3, .., :] -= u_euler[1, .., :] * u_gravity[3, .., :]
        @views @. du_euler[4, .., :] -= (u_euler[2, .., :] * u_gravity[2, .., :] +
                                         u_euler[3, .., :] * u_gravity[3, .., :])
    elseif ndims(semi_euler) == 3
        @views @. du_euler[2, .., :] -= u_euler[1, .., :] * u_gravity[2, .., :]
        @views @. du_euler[3, .., :] -= u_euler[1, .., :] * u_gravity[3, .., :]
        @views @. du_euler[4, .., :] -= u_euler[1, .., :] * u_gravity[4, .., :]
        @views @. du_euler[5, .., :] -= (u_euler[2, .., :] * u_gravity[2, .., :] +
                                         u_euler[3, .., :] * u_gravity[3, .., :] +
                                         u_euler[4, .., :] * u_gravity[4, .., :])
    else
        error("Number of dimensions $(ndims(semi_euler)) not supported.")
    end

    runtime = time_ns() - time_start
    put!(semi.performance_counter, runtime)

    return nothing
end

# TODO: Taal refactor, add some callbacks or so within the gravity update to allow investigating/optimizing it
function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode)
    @unpack semi_euler, semi_gravity, parameters, gravity_counter, cache = semi

    # Can be changed by AMR
    resize!(cache.du_ode, length(cache.u_ode))
    resize!(cache.u_tmp1_ode, length(cache.u_ode))
    resize!(cache.u_tmp2_ode, length(cache.u_ode))

    u_euler = wrap_array(u_ode, semi_euler)
    u_gravity = wrap_array(cache.u_ode, semi_gravity)
    du_gravity = wrap_array(cache.du_ode, semi_gravity)

    # set up main loop
    finalstep = false
    @unpack n_iterations_max, cfl, resid_tol, timestep_gravity = parameters
    iter = 0
    t = zero(real(semi_gravity.solver))

    # iterate gravity solver until convergence or maximum number of iterations are reached
    @unpack equations = semi_gravity
    while !finalstep
        dt = @trixi_timeit timer() "calculate dt" begin
            cfl * max_dt(u_gravity, t, semi_gravity.mesh,
                   have_constant_speed(equations), equations,
                   semi_gravity.solver, semi_gravity.cache)
        end

        # evolve solution by one pseudo-time step
        time_start = time_ns()
        timestep_gravity(cache, u_euler, t, dt, parameters, semi_gravity)
        runtime = time_ns() - time_start
        put!(gravity_counter, runtime)

        # update iteration counter
        iter += 1
        t += dt

        # check if we reached the maximum number of iterations
        if n_iterations_max > 0 && iter >= n_iterations_max
            @warn "Max iterations reached: Gravity solver failed to converge!" residual=maximum(abs,
                                                                                                @views du_gravity[1,
                                                                                                                  ..,
                                                                                                                  :]) t=t dt=dt
            finalstep = true
        end

        # this is an absolute tolerance check
        if maximum(abs, @views du_gravity[1, .., :]) <= resid_tol
            finalstep = true
        end
    end

    return nothing
end

# Integrate gravity solver for 2N-type low-storage schemes
function timestep_gravity_2N!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                              a, b, c)
    G = gravity_parameters.gravitational_constant
    rho0 = gravity_parameters.background_density
    grav_scale = -4.0 * pi * G

    @unpack u_ode, du_ode, u_tmp1_ode = cache
    u_tmp1_ode .= zero(eltype(u_tmp1_ode))
    du_gravity = wrap_array(du_ode, semi_gravity)
    for stage in eachindex(c)
        t_stage = t + dt * c[stage]

        # rhs! has the source term for the harmonic problem
        # We don't need a `@trixi_timeit timer() "rhs!"` here since that's already
        # included in the `rhs!` call.
        rhs!(du_ode, u_ode, semi_gravity, t_stage)

        # Source term: Jeans instability OR coupling convergence test OR blast wave
        # put in gravity source term proportional to Euler density
        # OBS! subtract off the background density ρ_0 (spatial mean value)
        @views @. du_gravity[1, .., :] += grav_scale * (u_euler[1, .., :] - rho0)

        a_stage = a[stage]
        b_stage_dt = b[stage] * dt
        @trixi_timeit timer() "Runge-Kutta step" begin
            @threaded for idx in eachindex(u_ode)
                u_tmp1_ode[idx] = du_ode[idx] - u_tmp1_ode[idx] * a_stage
                u_ode[idx] += u_tmp1_ode[idx] * b_stage_dt
            end
        end
    end

    return nothing
end

function timestep_gravity_carpenter_kennedy_erk54_2N!(cache, u_euler, t, dt,
                                                      gravity_parameters, semi_gravity)
    # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
    a = SVector(0.0, 567301805773.0 / 1357537059087.0,
                2404267990393.0 / 2016746695238.0,
                3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0)
    b = SVector(1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
                1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
                2277821191437.0 / 14882151754819.0)
    c = SVector(0.0, 1432997174477.0 / 9575080441755.0,
                2526269341429.0 / 6820363962896.0,
                2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0)

    timestep_gravity_2N!(cache, u_euler, t, dt, gravity_parameters, semi_gravity, a, b,
                         c)
end

# Integrate gravity solver for 3S*-type low-storage schemes
function timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters,
                                  semi_gravity,
                                  gamma1, gamma2, gamma3, beta, delta, c)
    G = gravity_parameters.gravitational_constant
    rho0 = gravity_parameters.background_density
    grav_scale = -4 * G * pi

    @unpack u_ode, du_ode, u_tmp1_ode, u_tmp2_ode = cache
    u_tmp1_ode .= zero(eltype(u_tmp1_ode))
    u_tmp2_ode .= u_ode
    du_gravity = wrap_array(du_ode, semi_gravity)
    for stage in eachindex(c)
        t_stage = t + dt * c[stage]

        # rhs! has the source term for the harmonic problem
        # We don't need a `@trixi_timeit timer() "rhs!"` here since that's already
        # included in the `rhs!` call.
        rhs!(du_ode, u_ode, semi_gravity, t_stage)

        # Source term: Jeans instability OR coupling convergence test OR blast wave
        # put in gravity source term proportional to Euler density
        # OBS! subtract off the background density ρ_0 around which the Jeans instability is perturbed
        @views @. du_gravity[1, .., :] += grav_scale * (u_euler[1, .., :] - rho0)

        delta_stage = delta[stage]
        gamma1_stage = gamma1[stage]
        gamma2_stage = gamma2[stage]
        gamma3_stage = gamma3[stage]
        beta_stage_dt = beta[stage] * dt
        @trixi_timeit timer() "Runge-Kutta step" begin
            @threaded for idx in eachindex(u_ode)
                u_tmp1_ode[idx] += delta_stage * u_ode[idx]
                u_ode[idx] = (gamma1_stage * u_ode[idx] +
                              gamma2_stage * u_tmp1_ode[idx] +
                              gamma3_stage * u_tmp2_ode[idx] +
                              beta_stage_dt * du_ode[idx])
            end
        end
    end

    return nothing
end

function timestep_gravity_erk51_3Sstar!(cache, u_euler, t, dt, gravity_parameters,
                                        semi_gravity)
    # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
    # and examples/parameters_hypdiff_lax_friedrichs.toml
    # 5 stages, order 1
    gamma1 = SVector(0.0000000000000000E+00, 5.2910412316555866E-01,
                     2.8433964362349406E-01, -1.4467571130907027E+00,
                     7.5592215948661057E-02)
    gamma2 = SVector(1.0000000000000000E+00, 2.6366970460864109E-01,
                     3.7423646095836322E-01, 7.8786901832431289E-01,
                     3.7754129043053775E-01)
    gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,
                     0.0000000000000000E+00, 8.0043329115077388E-01,
                     1.3550099149374278E-01)
    beta = SVector(1.9189497208340553E-01, 5.4506406707700059E-02,
                   1.2103893164085415E-01, 6.8582252490550921E-01,
                   8.7914657211972225E-01)
    delta = SVector(1.0000000000000000E+00, 7.8593091509463076E-01,
                    1.2639038717454840E-01, 1.7726945920209813E-01,
                    0.0000000000000000E+00)
    c = SVector(0.0000000000000000E+00, 1.9189497208340553E-01, 1.9580448818599061E-01,
                2.4241635859769023E-01, 5.0728347557552977E-01)

    timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                             gamma1, gamma2, gamma3, beta, delta, c)
end

function timestep_gravity_erk52_3Sstar!(cache, u_euler, t, dt, gravity_parameters,
                                        semi_gravity)
    # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
    # and examples/parameters_hypdiff_lax_friedrichs.toml
    # 5 stages, order 2
    gamma1 = SVector(0.0000000000000000E+00, 5.2656474556752575E-01,
                     1.0385212774098265E+00, 3.6859755007388034E-01,
                     -6.3350615190506088E-01)
    gamma2 = SVector(1.0000000000000000E+00, 4.1892580153419307E-01,
                     -2.7595818152587825E-02, 9.1271323651988631E-02,
                     6.8495995159465062E-01)
    gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,
                     0.0000000000000000E+00, 4.1301005663300466E-01,
                     -5.4537881202277507E-03)
    beta = SVector(4.5158640252832094E-01, 7.5974836561844006E-01,
                   3.7561630338850771E-01, 2.9356700007428856E-02,
                   2.5205285143494666E-01)
    delta = SVector(1.0000000000000000E+00, 1.3011720142005145E-01,
                    2.6579275844515687E-01, 9.9687218193685878E-01,
                    0.0000000000000000E+00)
    c = SVector(0.0000000000000000E+00, 4.5158640252832094E-01, 1.0221535725056414E+00,
                1.4280257701954349E+00, 7.1581334196229851E-01)

    timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                             gamma1, gamma2, gamma3, beta, delta, c)
end

function timestep_gravity_erk53_3Sstar!(cache, u_euler, t, dt, gravity_parameters,
                                        semi_gravity)
    # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
    # and examples/parameters_hypdiff_lax_friedrichs.toml
    # 5 stages, order 3
    gamma1 = SVector(0.0000000000000000E+00, 6.9362208054011210E-01,
                     9.1364483229179472E-01, 1.3129305757628569E+00,
                     -1.4615811339132949E+00)
    gamma2 = SVector(1.0000000000000000E+00, 1.3224582239681788E+00,
                     2.4213162353103135E-01, -3.8532017293685838E-01,
                     1.5603355704723714E+00)
    gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,
                     0.0000000000000000E+00, 3.8306787039991996E-01,
                     -3.5683121201711010E-01)
    beta = SVector(8.4476964977404881E-02, 3.0834660698015803E-01,
                   3.2131664733089232E-01, 2.8783574345390539E-01,
                   8.2199204703236073E-01)
    delta = SVector(1.0000000000000000E+00, -7.6832695815481578E-01,
                    1.2497251501714818E-01, 1.4496404749796306E+00,
                    0.0000000000000000E+00)
    c = SVector(0.0000000000000000E+00, 8.4476964977404881E-02, 2.8110631488732202E-01,
                5.7093842145029405E-01, 7.2999896418559662E-01)

    timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                             gamma1, gamma2, gamma3, beta, delta, c)
end

# TODO: Taal decide, where should specific parts like these be?
@inline function save_solution_file(u_ode, t, dt, iter,
                                    semi::SemidiscretizationEulerGravity,
                                    solution_callback,
                                    element_variables = Dict{Symbol, Any}();
                                    system = "")
    # If this is called already as part of a multi-system setup (i.e., system is non-empty),
    # we build a combined system name
    if !isempty(system)
        system_euler = system * "_euler"
        system_gravity = system * "_gravity"
    else
        system_euler = "euler"
        system_gravity = "gravity"
    end

    u_euler = wrap_array_native(u_ode, semi.semi_euler)
    filename_euler = save_solution_file(u_euler, t, dt, iter,
                                        mesh_equations_solver_cache(semi.semi_euler)...,
                                        solution_callback, element_variables,
                                        system = system_euler)

    u_gravity = wrap_array_native(semi.cache.u_ode, semi.semi_gravity)
    filename_gravity = save_solution_file(u_gravity, t, dt, iter,
                                          mesh_equations_solver_cache(semi.semi_gravity)...,
                                          solution_callback, element_variables,
                                          system = system_gravity)

    return filename_euler, filename_gravity
end

@inline function (amr_callback::AMRCallback)(u_ode,
                                             semi::SemidiscretizationEulerGravity,
                                             t, iter; kwargs...)
    passive_args = ((semi.cache.u_ode,
                     mesh_equations_solver_cache(semi.semi_gravity)...),)
    amr_callback(u_ode, mesh_equations_solver_cache(semi.semi_euler)..., semi, t, iter;
                 kwargs..., passive_args = passive_args)
end
end # @muladd
