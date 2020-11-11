
# Integrate solution by repeatedly calling the rhs! method on the solver solution.
# function timestep_XYZ!(solver::AbstractSolver, t, dt)


"""
    timestep_2N!(solver, t, dt, a, b, c)

Perform one timestep using an explicit Runge-Kutta method of the low-storage
class 2N of Williamson.
"""
@inline function timestep_2N!(mesh, solver, t, dt, a, b, c)
  for stage in eachindex(c)
    t_stage = t + dt * c[stage]
    ndims_ = parameter("ndims")::Int
    if ndims_ == 3
      @timeit timer() "rhs" rhs!(solver, t_stage)
    else
      @timeit timer() "rhs" rhs!(mesh, solver, t_stage)
    end

    a_stage    = a[stage]
    b_stage_dt = b[stage] * dt
    @timeit timer() "Runge-Kutta step" begin
      Threads.@threads for i in eachindex(solver.elements.u)
        solver.elements.u_tmp2[i] = solver.elements.u_t[i] - solver.elements.u_tmp2[i] * a_stage
        solver.elements.u[i] += solver.elements.u_tmp2[i] * b_stage_dt
      end
    end
  end
  apply_positivity_preserving_limiter!(solver)
end

"""
    timestep_carpenter_kennedy_erk54_2N!(solver::AbstractSolver, t, dt)

Carpenter, Kennedy (1994) Fourth order 2N storage RK schemes, Solution 3
"""
function timestep_carpenter_kennedy_erk54_2N!(mesh::TreeMesh, solver::AbstractSolver, t, dt)
  a = @SVector [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = @SVector [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
       2277821191437.0 / 14882151754819.0]
  c = @SVector [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]

  timestep_2N!(mesh, solver, t, dt, a, b, c)
end

"""
    timestep_carpenter_kennedy_erk43_2N!(solver::AbstractSolver, t, dt)

Carpenter, Kennedy (1994) Third order 2N storage RK schemes with error control
"""
function timestep_carpenter_kennedy_erk43_2N!(mesh::TreeMesh, solver::AbstractSolver, t, dt)
  a = @SVector [0, 756391 / 934407, 36441873 / 15625000, 1953125 / 1085297]
  b = @SVector [8 / 141, 6627 / 2000, 609375 / 1085297, 198961 / 526383]
  c = @SVector [0, 8 / 141, 86 / 125, 1]

  timestep_2N!(mesh, solver, t, dt, a, b, c)
end


"""
    timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)

Perform one timestep using an explicit Runge-Kutta method of the low-storage
class 3Sstar of Ketcheson.
"""
@inline function timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)
  solver.elements.u_tmp2 .= zero(eltype(solver.elements.u_tmp2))
  solver.elements.u_tmp3 .= solver.elements.u
  for stage in eachindex(c)
    t_stage = t + dt * c[stage]
    @timeit timer() "rhs" rhs!(solver, t_stage)

    delta_stage   = delta[stage]
    gamma1_stage  = gamma1[stage]
    gamma2_stage  = gamma2[stage]
    gamma3_stage  = gamma3[stage]
    beta_stage_dt = beta[stage] * dt
    @timeit timer() "Runge-Kutta step" begin
      Threads.@threads for i in eachindex(solver.elements.u)
        solver.elements.u_tmp2[i] += delta_stage * solver.elements.u[i]
        solver.elements.u[i]       = (gamma1_stage * solver.elements.u[i] +
                                      gamma2_stage * solver.elements.u_tmp2[i] +
                                      gamma3_stage * solver.elements.u_tmp3[i] +
                                      beta_stage_dt * solver.elements.u_t[i])
      end
    end
  end
end


"""
    timestep_hyp_diff_N3_erk52_3Sstar!(solver::AbstractSolver, t, dt)

Five stage, second-order acurate explicit Runge-Kutta scheme with stability region optimized for
the hyperbolic diffusion equation with LLF flux and polynomials of degree polydeg=3.
"""
function timestep_hyp_diff_N3_erk52_3Sstar!(solver::AbstractSolver, t, dt)
  # # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
  # # and examples/parameters_hyp_diff_llf.toml
  # # 5 stages, order 1
  # gamma1 = @SVector [0.0000000000000000E+00, 5.2910412316555866E-01, 2.8433964362349406E-01, -1.4467571130907027E+00, 7.5592215948661057E-02]
  # gamma2 = @SVector [1.0000000000000000E+00, 2.6366970460864109E-01, 3.7423646095836322E-01, 7.8786901832431289E-01, 3.7754129043053775E-01]
  # gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 8.0043329115077388E-01, 1.3550099149374278E-01]
  # beta   = @SVector [1.9189497208340553E-01, 5.4506406707700059E-02, 1.2103893164085415E-01, 6.8582252490550921E-01, 8.7914657211972225E-01]
  # delta  = @SVector [1.0000000000000000E+00, 7.8593091509463076E-01, 1.2639038717454840E-01, 1.7726945920209813E-01, 0.0000000000000000E+00]
  # c      = @SVector [0.0000000000000000E+00, 1.9189497208340553E-01, 1.9580448818599061E-01, 2.4241635859769023E-01, 5.0728347557552977E-01]
  # 5 stages, order 2
  gamma1 = @SVector [0.0000000000000000E+00, 5.2656474556752575E-01, 1.0385212774098265E+00, 3.6859755007388034E-01, -6.3350615190506088E-01]
  gamma2 = @SVector [1.0000000000000000E+00, 4.1892580153419307E-01, -2.7595818152587825E-02, 9.1271323651988631E-02, 6.8495995159465062E-01]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 4.1301005663300466E-01, -5.4537881202277507E-03]
  beta   = @SVector [4.5158640252832094E-01, 7.5974836561844006E-01, 3.7561630338850771E-01, 2.9356700007428856E-02, 2.5205285143494666E-01]
  delta  = @SVector [1.0000000000000000E+00, 1.3011720142005145E-01, 2.6579275844515687E-01, 9.9687218193685878E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 4.5158640252832094E-01, 1.0221535725056414E+00, 1.4280257701954349E+00, 7.1581334196229851E-01]
  # # 5 stages, order 3
  # gamma1 = @SVector [0.0000000000000000E+00, 6.9362208054011210E-01, 9.1364483229179472E-01, 1.3129305757628569E+00, -1.4615811339132949E+00]
  # gamma2 = @SVector [1.0000000000000000E+00, 1.3224582239681788E+00, 2.4213162353103135E-01, -3.8532017293685838E-01, 1.5603355704723714E+00]
  # gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 3.8306787039991996E-01, -3.5683121201711010E-01]
  # beta   = @SVector [8.4476964977404881E-02, 3.0834660698015803E-01, 3.2131664733089232E-01, 2.8783574345390539E-01, 8.2199204703236073E-01]
  # delta  = @SVector [1.0000000000000000E+00, -7.6832695815481578E-01, 1.2497251501714818E-01, 1.4496404749796306E+00, 0.0000000000000000E+00]
  # c      = @SVector [0.0000000000000000E+00, 8.4476964977404881E-02, 2.8110631488732202E-01, 5.7093842145029405E-01, 7.2999896418559662E-01]

  timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)
end


"""
    timestep_parsani_ketcheson_deconinck_erk94_3Sstar!(solver::AbstractSolver, t, dt)

Parsani, Ketcheson, Deconinck (2013)
  Optimized explicit RK schemes for the spectral difference method applied to wave propagation problems
[DOI: 10.1137/120885899](https://doi.org/10.1137/120885899)
"""
function timestep_parsani_ketcheson_deconinck_erk94_3Sstar!(solver::AbstractSolver, t, dt)
  gamma1 = @SVector [0.0000000000000000E+00, -4.6556413837561301E+00, -7.7202649689034453E-01, -4.0244202720632174E+00, -2.1296873883702272E-02, -2.4350219407769953E+00, 1.9856336960249132E-02, -2.8107894116913812E-01, 1.6894354373677900E-01]
  gamma2 = @SVector [1.0000000000000000E+00, 2.4992627683300688E+00, 5.8668202764174726E-01, 1.2051419816240785E+00, 3.4747937498564541E-01, 1.3213458736302766E+00, 3.1196363453264964E-01, 4.3514189245414447E-01, 2.3596980658341213E-01]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 7.6209857891449362E-01, -1.9811817832965520E-01, -6.2289587091629484E-01, -3.7522475499063573E-01, -3.3554373281046146E-01, -4.5609629702116454E-02]
  beta   = @SVector [2.8363432481011769E-01, 9.7364980747486463E-01, 3.3823592364196498E-01, -3.5849518935750763E-01, -4.1139587569859462E-03, 1.4279689871485013E+00, 1.8084680519536503E-02, 1.6057708856060501E-01, 2.9522267863254809E-01]
  delta  = @SVector [1.0000000000000000E+00, 1.2629238731608268E+00, 7.5749675232391733E-01, 5.1635907196195419E-01, -2.7463346616574083E-02, -4.3826743572318672E-01, 1.2735870231839268E+00, -6.2947382217730230E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 2.8363432481011769E-01, 5.4840742446661772E-01, 3.6872298094969475E-01, -6.8061183026103156E-01, 3.5185265855105619E-01, 1.6659419385562171E+00, 9.7152778807463247E-01, 9.0515694340066954E-01]

  timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)
end


"""
    timestep_parsani_ketcheson_deconinck_erk32_3Sstar!(solver::AbstractSolver, t, dt)

Parsani, Ketcheson, Deconinck (2013)
  Optimized explicit RK schemes for the spectral difference method applied to wave propagation problems
[DOI: 10.1137/120885899](https://doi.org/10.1137/120885899)
"""
function timestep_parsani_ketcheson_deconinck_erk32_3Sstar!(solver::AbstractSolver, t, dt)
  gamma1 = @SVector [0.0000000000000000E+00, -1.2664395576322218E-01, 1.1426980685848858E+00]
  gamma2 = @SVector [1.0000000000000000E+00, 6.5427782599406470E-01, -8.2869287683723744E-02]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00]
  beta   = @SVector [7.2366074728360086E-01, 3.4217876502651023E-01, 3.6640216242653251E-01]
  delta  = @SVector [1.0000000000000000E+00, 7.2196567116037724E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 7.2366074728360086E-01, 5.9236433182015646E-01]

  timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)
end

# Add implemenations for coupled Euler-gravity simulations
include("timedisc_euler_gravity.jl")
