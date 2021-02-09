
# Abstract base type for time integration schemes of storage class `3S*`
abstract type SimpleAlgorithm3Sstar end


"""
    HypDiffN3Erk3Sstar52()

Five stage, second-order acurate explicit Runge-Kutta scheme with stability region optimized for
the hyperbolic diffusion equation with LLF flux and polynomials of degree polydeg=3.
"""
struct HypDiffN3Erk3Sstar52 <: SimpleAlgorithm3Sstar
  gamma1::SVector{5, Float64}
  gamma2::SVector{5, Float64}
  gamma3::SVector{5, Float64}
  beta::SVector{5, Float64}
  delta::SVector{5, Float64}
  c::SVector{5, Float64}

  function HypDiffN3Erk3Sstar52()
    gamma1 = SVector(0.0000000000000000E+00, 5.2656474556752575E-01,  1.0385212774098265E+00, 3.6859755007388034E-01, -6.3350615190506088E-01)
    gamma2 = SVector(1.0000000000000000E+00, 4.1892580153419307E-01, -2.7595818152587825E-02, 9.1271323651988631E-02,  6.8495995159465062E-01)
    gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,  0.0000000000000000E+00, 4.1301005663300466E-01, -5.4537881202277507E-03)
    beta   = SVector(4.5158640252832094E-01, 7.5974836561844006E-01,  3.7561630338850771E-01, 2.9356700007428856E-02,  2.5205285143494666E-01)
    delta  = SVector(1.0000000000000000E+00, 1.3011720142005145E-01,  2.6579275844515687E-01, 9.9687218193685878E-01,  0.0000000000000000E+00)
    c      = SVector(0.0000000000000000E+00, 4.5158640252832094E-01,  1.0221535725056414E+00, 1.4280257701954349E+00,  7.1581334196229851E-01)

    new(gamma1, gamma2, gamma3, beta, delta, c)
  end
end

struct HypDiffN3Erk3SstarTest{Steps} <: SimpleAlgorithm3Sstar
  gamma1::SVector{Steps, Float64}
  gamma2::SVector{Steps, Float64}
  gamma3::SVector{Steps, Float64}
  beta  ::SVector{Steps, Float64}
  delta ::SVector{Steps, Float64}
  c     ::SVector{Steps, Float64}

  function HypDiffN3Erk3SstarTest{Steps}() where {Steps}
    # 2 stages, order 1, erk-1-2_2021-01-21T16-59-55.txt
    # gamma1 = SVector(0.0000000000000000E+00, -1.2180814892100231E+00)
    # gamma2 = SVector(1.0000000000000000E+00,  2.2180814892100231E+00)
    # gamma3 = SVector(0.0000000000000000E+00,  0.0000000000000000E+00)
    # beta   = SVector(2.5570794725413060E-01,  1.3114731171941489E+00)
    # delta  = SVector(1.0000000000000000E+00,  0.0000000000000000E+00)
    # c      = SVector(0.0000000000000000E+00,  2.5570794725413060E-01)
    if Steps == 3
      # 3 stages, order 1, 3Sstar-1-3_2021-01-21T18-19-08.txt
      # This scheme can be between ca. 5% and 20% better than `timestep_gravity_erk52_3Sstar!`
      # for EOC and Jeans, depending on the tolerances... However, it does not seem to be better
      # for Sedov...
      gamma1 = SVector(0.0000000000000000E+00, 5.3542666596047617E-01, 9.1410889739925583E-01)
      gamma2 = SVector(1.0000000000000000E+00, 4.3591819397582626E-01, 8.0593291910989059E-02)
      gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00)
      beta   = SVector(5.1434308417699570E-01, 5.0548475589200048E-01, 2.6999513995934876E-01)
      delta  = SVector(1.0000000000000000E+00, 6.5735132095190080E-02, 0.0000000000000000E+00)
      c      = SVector(0.0000000000000000E+00, 5.1434308417699570E-01, 7.9561633173060387E-01)
      # # 3 stages, order 1, 3Sstar-1-3_2021-02-09T15-11-17.txt
      # # This scheme is created via a matrix Chebyshev polynomial for
      # # initial_condition_harmonic_nonperiodic, solver = DGSEM(3, flux_godunov), initial_refinement_level=1
      # # For simple hyp. diff. tests, it appears to be a bit better than the method above
      # gamma1 = SVector(0.0000000000000000E+00, 1.8049224575489586E-01, 6.2802664655218099E-01)
      # gamma2 = SVector(1.0000000000000000E+00, 4.6411714543807780E-01, 2.1066208353361934E-01)
      # gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00)
      # beta   = SVector(4.1601970882141232E-01, 5.9131800779192267E-01, 4.2151696977108938E-01)
      # delta  = SVector(1.0000000000000000E+00, 7.6573471223859868E-01, 0.0000000000000000E+00)
      # c      = SVector(0.0000000000000000E+00, 4.1601970882141232E-01, 8.1425583690916081E-01)
    elseif Steps == 4
      # 4 stages, order 1, 3Sstar-1-4_2021-01-21T18-19-22.txt
      gamma1 = SVector(0.0000000000000000E+00, -5.6625536579338731E-01,  1.3418587031785676E+00, 8.8890148834863436E-01)
      gamma2 = SVector(1.0000000000000000E+00,  1.1536881793250324E+00, -1.8570628751004167E-01, 3.8337868078920548E-02)
      gamma3 = SVector(0.0000000000000000E+00,  0.0000000000000000E+00,  0.0000000000000000E+00, 4.0523982162793015E-02)
      beta   = SVector(2.3171237092062003E-01,  3.2627637023876249E-01,  1.9590115587227472E-01, 5.0746738742196784E-01)
      delta  = SVector(1.0000000000000000E+00,  3.5760718872037695E-01,  4.8324972443116271E-01, 0.0000000000000000E+00)
      c      = SVector(0.0000000000000000E+00,  2.3171237092062003E-01,  2.9066491782488579E-01, 5.4445940944117321E-01)
      # # 4 stages, order 2, 3Sstar-2-4_2021-01-21T18-22-47.txt
      # gamma1 = SVector(0.0000000000000000E+00, 3.9242586618503339E-01,  1.0035329389428556E+00,  5.5592551752986541E-01)
      # gamma2 = SVector(1.0000000000000000E+00, 3.1722869401108084E-01, -1.6915015356303413E-03,  2.1306582216797171E-01)
      # gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,  0.0000000000000000E+00, -9.4346440879594987E-04)
      # beta   = SVector(2.9330285434493913E-01, 3.3265933704162054E-01,  2.1171710030945143E-01,  5.0844675353885938E-01)
      # delta  = SVector(1.0000000000000000E+00, 9.1525591879070056E-01,  1.7338477614017425E-01,  0.0000000000000000E+00)
      # c      = SVector(0.0000000000000000E+00, 2.9330285434493913E-01,  5.3291810995299238E-01,  7.4590760404926193E-01)
    elseif Steps == 5
      # 5 stages, order 1, 3Sstar-1-5_2021-01-22T06-47-41.txt
      gamma1 = SVector(0.0000000000000000E+00, -1.8802729519221351E-01, 4.6604651511432649E-01,  8.8394183350193534E-01, -6.5068141753484565E-01)
      gamma2 = SVector(1.0000000000000000E+00,  5.4134480367690074E-01, 2.0790247136947018E-01, -7.5830656883294466E-04,  6.2410224487791299E-01)
      gamma3 = SVector(0.0000000000000000E+00,  0.0000000000000000E+00, 0.0000000000000000E+00,  1.1833886026287863E-01, -2.2637748845630642E-01)
      beta   = SVector(1.4639181558742428E-01,  3.1071563744231162E-01, 8.7907721397119457E-02,  1.7847768337323872E-01,  1.0178754551387090E+00)
      delta  = SVector(1.0000000000000000E+00,  1.1945852017474661E+00, 3.7370309886040687E-01,  4.3932611096717716E-01,  0.0000000000000000E+00)
      c      = SVector(0.0000000000000000E+00,  1.4639181558742428E-01, 3.7785900436065667E-01,  3.2972235839127506E-01,  4.6958353540042619E-01)
    elseif Steps == 7
      # 7 stages, order 2
      gamma1 = SVector(0.0000000000000000E+00, 7.4645855541196471E-01,  1.1933706288014441E+00, 4.4882143764278509E-01,  1.2597816001927011E+00, -1.9395359618180008E-01, -2.0529989434528106E-01)
      gamma2 = SVector(1.0000000000000000E+00, 1.9604567859122732E-01, -9.4228044406896444E-02, 1.8274409185201260E-01, -5.0215951766262042E-02,  1.6864474841417218E-01,  7.7591329451069480E-01)
      gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00,  0.0000000000000000E+00, 1.3273971095845494E-01, -1.2014683669108431E-01,  6.8162040983132111E-01, -1.1518807130939930E+00)
      beta   = SVector(2.9710300213012131E-01, 2.8004654286613695E-01,  1.8558682604908017E-01, 4.7849858779738258E-01,  4.6699827604810107E-01, -1.0894983405033070E-02, -2.0239599603544003E-01)
      delta  = SVector(1.0000000000000000E+00, 2.9327739540075143E-01,  7.5887841467240658E-01, 2.3759729306746027E-01,  4.9093228845702752E-01,  2.5725792515626500E-01,  0.0000000000000000E+00)
      c = zero(gamma1)
    end
    # # xxx stages, order xxx
    # gamma1 = SVector(0.0000000000000000E+00, )
    # gamma2 = SVector(1.0000000000000000E+00, )
    # gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, )
    # beta   = SVector()
    # delta  = SVector(1.0000000000000000E+00, , 0.0000000000000000E+00)
    # c      = SVector(0.0000000000000000E+00, )
    # c      = zero(gamma1)

    new(gamma1, gamma2, gamma3, beta, delta, c)
  end
end


"""
    ParsaniKetchesonDeconinck3Sstar94()

Parsani, Ketcheson, Deconinck (2013)
  Optimized explicit RK schemes for the spectral difference method applied to wave propagation problems
[DOI: 10.1137/120885899](https://doi.org/10.1137/120885899)
"""
struct ParsaniKetchesonDeconinck3Sstar94 <: SimpleAlgorithm3Sstar
  gamma1::SVector{9, Float64}
  gamma2::SVector{9, Float64}
  gamma3::SVector{9, Float64}
  beta::SVector{9, Float64}
  delta::SVector{9, Float64}
  c::SVector{9, Float64}

  function ParsaniKetchesonDeconinck3Sstar94()
    gamma1 = SVector(0.0000000000000000E+00, -4.6556413837561301E+00, -7.7202649689034453E-01, -4.0244202720632174E+00, -2.1296873883702272E-02, -2.4350219407769953E+00, 1.9856336960249132E-02, -2.8107894116913812E-01, 1.6894354373677900E-01)
    gamma2 = SVector(1.0000000000000000E+00, 2.4992627683300688E+00, 5.8668202764174726E-01, 1.2051419816240785E+00, 3.4747937498564541E-01, 1.3213458736302766E+00, 3.1196363453264964E-01, 4.3514189245414447E-01, 2.3596980658341213E-01)
    gamma3 = SVector(0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 7.6209857891449362E-01, -1.9811817832965520E-01, -6.2289587091629484E-01, -3.7522475499063573E-01, -3.3554373281046146E-01, -4.5609629702116454E-02)
    beta   = SVector(2.8363432481011769E-01, 9.7364980747486463E-01, 3.3823592364196498E-01, -3.5849518935750763E-01, -4.1139587569859462E-03, 1.4279689871485013E+00, 1.8084680519536503E-02, 1.6057708856060501E-01, 2.9522267863254809E-01)
    delta  = SVector(1.0000000000000000E+00, 1.2629238731608268E+00, 7.5749675232391733E-01, 5.1635907196195419E-01, -2.7463346616574083E-02, -4.3826743572318672E-01, 1.2735870231839268E+00, -6.2947382217730230E-01, 0.0000000000000000E+00)
    c      = SVector(0.0000000000000000E+00, 2.8363432481011769E-01, 5.4840742446661772E-01, 3.6872298094969475E-01, -6.8061183026103156E-01, 3.5185265855105619E-01, 1.6659419385562171E+00, 9.7152778807463247E-01, 9.0515694340066954E-01)

    new(gamma1, gamma2, gamma3, beta, delta, c)
  end
end


"""
    ParsaniKetchesonDeconinck3Sstar32()

Parsani, Ketcheson, Deconinck (2013)
  Optimized explicit RK schemes for the spectral difference method applied to wave propagation problems
[DOI: 10.1137/120885899](https://doi.org/10.1137/120885899)
"""
struct ParsaniKetchesonDeconinck3Sstar32 <: SimpleAlgorithm3Sstar
  gamma1::SVector{3, Float64}
  gamma2::SVector{3, Float64}
  gamma3::SVector{3, Float64}
  beta::SVector{3, Float64}
  delta::SVector{3, Float64}
  c::SVector{3, Float64}

  function ParsaniKetchesonDeconinck3Sstar32()
    gamma1 = SVector(0.0000000000000000E+00, -1.2664395576322218E-01,  1.1426980685848858E+00)
    gamma2 = SVector(1.0000000000000000E+00,  6.5427782599406470E-01, -8.2869287683723744E-02)
    gamma3 = SVector(0.0000000000000000E+00,  0.0000000000000000E+00,  0.0000000000000000E+00)
    beta   = SVector(7.2366074728360086E-01,  3.4217876502651023E-01,  3.6640216242653251E-01)
    delta  = SVector(1.0000000000000000E+00,  7.2196567116037724E-01,  0.0000000000000000E+00)
    c      = SVector(0.0000000000000000E+00,  7.2366074728360086E-01,  5.9236433182015646E-01)

    new(gamma1, gamma2, gamma3, beta, delta, c)
  end
end


mutable struct SimpleIntegrator3SstarOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function SimpleIntegrator3SstarOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  SimpleIntegrator3SstarOptions{typeof(callback)}(
    callback, false, Inf, maxiters, [last(tspan)])
end

mutable struct SimpleIntegrator3Sstar{RealT<:Real, uType, Params, Sol, Alg, SimpleIntegrator3SstarOptions}
  u::uType #
  du::uType
  u_tmp1::uType
  u_tmp2::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time step (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  alg::Alg
  opts::SimpleIntegrator3SstarOptions
  finalstep::Bool # added for convenience
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::T;
               dt, callback=nothing, kwargs...) where {T<:SimpleAlgorithm3Sstar}
  u = copy(ode.u0)
  du = similar(u)
  u_tmp1 = similar(u)
  u_tmp2 = similar(u)
  t = first(ode.tspan)
  iter = 0
  integrator = SimpleIntegrator3Sstar(u, du, u_tmp1, u_tmp2, t, dt, zero(dt), iter, ode.p,
                  (prob=ode,), alg,
                  SimpleIntegrator3SstarOptions(callback, ode.tspan; kwargs...), false)

  # initialize callbacks
  if callback isa CallbackSet
    for cb in callback.continuous_callbacks
      error("unsupported")
    end
    for cb in callback.discrete_callbacks
      cb.initialize(cb, integrator.u, integrator.t, integrator)
    end
  elseif !isnothing(callback)
    error("unsupported")
  end

  solve!(integrator)
end

function solve!(integrator::SimpleIntegrator3Sstar)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan)
  callbacks = integrator.opts.callback

  integrator.finalstep = false
  @timeit_debug timer() "main loop" while !integrator.finalstep
    if isnan(integrator.dt)
      error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
      integrator.dt = t_end - integrator.t
      terminate!(integrator)
    end

    # one time step
    integrator.u_tmp1 .= zero(eltype(integrator.u_tmp1))
    integrator.u_tmp2 .= integrator.u
    for stage in eachindex(alg.c)
      t_stage = integrator.t + integrator.dt * alg.c[stage]
      prob.f(integrator.du, integrator.u, prob.p, t_stage)

      delta_stage   = alg.delta[stage]
      gamma1_stage  = alg.gamma1[stage]
      gamma2_stage  = alg.gamma2[stage]
      gamma3_stage  = alg.gamma3[stage]
      beta_stage_dt = alg.beta[stage] * integrator.dt
      @timeit_debug timer() "Runge-Kutta step" begin
        @threaded for i in eachindex(integrator.u)
          integrator.u_tmp1[i] += delta_stage * integrator.u[i]
          integrator.u[i]       = (gamma1_stage * integrator.u[i] +
                                   gamma2_stage * integrator.u_tmp1[i] +
                                   gamma3_stage * integrator.u_tmp2[i] +
                                   beta_stage_dt * integrator.du[i])
        end
      end
    end
    integrator.iter += 1
    integrator.t += integrator.dt

    # handle callbacks
    if callbacks isa CallbackSet
      for cb in callbacks.discrete_callbacks
        if cb.condition(integrator.u, integrator.t, integrator)
          cb.affect!(integrator)
        end
      end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
      @warn "Interrupted. Larger maxiters is needed."
      terminate!(integrator)
    end
  end

  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleIntegrator3Sstar) = integrator.du
get_tmp_cache(integrator::SimpleIntegrator3Sstar) = (integrator.u_tmp1, integrator.u_tmp2)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleIntegrator3Sstar, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleIntegrator3Sstar, dt)
  integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::SimpleIntegrator3Sstar)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR
function Base.resize!(integrator::SimpleIntegrator3Sstar, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp1, new_size)
  resize!(integrator.u_tmp2, new_size)
end
