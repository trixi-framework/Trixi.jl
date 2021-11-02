# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


  @doc raw"""
      CompressibleEulerMultichemistryEquations2D(; gammas, gas_constants)
  !!! warning "Experimental code"
      This system of equations is experimental and can change any time.
  Multichemistry version of the compressible Euler equations
  ```math
  \partial t
  \begin{pmatrix}
  \rho v_1 \\ \rho v_2 \\ E \\ \rho_1 \\ \rho_2 \\ \vdots \\ \rho_{n}
  \end{pmatrix}
  +
  \partial x
  \begin{pmatrix}
  \rho v_1^2 + p \\ \rho v_1 v_2 \\ (E+p) v_1 \\ \rho_1 v_1 \\ \rho_2 v_1 \\ \vdots \\ \rho_{n} v_1
  \end{pmatrix}
  +
  \partial y
  \begin{pmatrix}
  \rho v_1 v_2 \\ \rho v_2^2 + p \\ (E+p) v_2 \\ \rho_1 v_2 \\ \rho_2 v_2 \\ \vdots \\ \rho_{n} v_2
  \end{pmatrix}
  =
  \begin{pmatrix}
  0 \\ 0 \\ 0 \\ 0 \\ 0 \\ \vdots \\ 0
  \end{pmatrix}
  ```
  for calorically perfect gas in two space dimensions.
  In case of more than one component, the specific heat ratios `gammas` and the gas constants
  `gas_constants` in [kJ/(kg*K)] should be passed as tuples, e.g., `gammas=(1.4, 1.667)`.
  The remaining variables like the specific heats at constant volume 'cv' or the specific heats at
  constant pressure 'cp' are then calculated considering a calorically perfect gas.
  """
  mutable struct CompressibleEulerMultichemistryEquations2D{NVARS, NCOMP, RealT<:Real} <: AbstractCompressibleEulerMultichemistryEquations{2, NVARS, NCOMP}
    gammas            ::SVector{NCOMP, RealT}
    gas_constants     ::SVector{NCOMP, RealT}
    cv                ::SVector{NCOMP, RealT}
    cp                ::SVector{NCOMP, RealT}
    heat_of_formations::SVector{NCOMP, RealT}
    t_old             ::RealT
    delta_t           ::RealT
  
    function CompressibleEulerMultichemistryEquations2D{NVARS, NCOMP, RealT}(gammas            ::SVector{NCOMP, RealT},
                                                                             gas_constants     ::SVector{NCOMP, RealT},
                                                                             heat_of_formations::SVector{NCOMP, RealT}) where {NVARS, NCOMP, RealT<:Real}
  
      NCOMP >= 1 || throw(DimensionMismatch("`gammas` and `gas_constants` have to be filled with at least one value"))
  
      cv = gas_constants ./ (gammas .- 1)
      cp = gas_constants + gas_constants ./ (gammas .- 1)

      krome_init() # Start Krome

      t_old = 0.0
      delta_t = 0.0
  
      new(gammas, gas_constants, cv, cp, heat_of_formations, t_old, delta_t)
    end
  end
  
  
  function CompressibleEulerMultichemistryEquations2D(; gammas, gas_constants, heat_of_formations)
  
    _gammas        = promote(gammas...)
    _gas_constants = promote(gas_constants...)
    _heat_of_formations = promote(heat_of_formations...)
    RealT          = promote_type(eltype(_gammas), eltype(_gas_constants), eltype(_heat_of_formations))
  
    NVARS = length(_gammas) + 3
    NCOMP = length(_gammas)
  
    __gammas        = SVector(map(RealT, _gammas))
    __gas_constants = SVector(map(RealT, _gas_constants))
    __heat_of_formations = SVector(map(RealT, _heat_of_formations))
  
    return CompressibleEulerMultichemistryEquations2D{NVARS, NCOMP, RealT}(__gammas, __gas_constants, __heat_of_formations)
  end
  
  
  @inline Base.real(::CompressibleEulerMultichemistryEquations2D{NVARS, NCOMP, RealT}) where {NVARS, NCOMP, RealT} = RealT
  
  
  function varnames(::typeof(cons2cons), equations::CompressibleEulerMultichemistryEquations2D)
  
    cons  = ("rho_v1", "rho_v2", "rho_e")
    rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (cons..., rhos...)
  end
  
  
  function varnames(::typeof(cons2prim), equations::CompressibleEulerMultichemistryEquations2D)
  
    prim  = ("v1", "v2", "p")
    rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (prim..., rhos...)
  end
  
  
  # Set initial conditions at physical location `x` for time `t`
  
  """
      initial_condition_convergence_test(x, t, equations::CompressibleEulerMultichemistryEquations2D)
  A smooth initial condition used for convergence tests in combination with
  [`source_terms_convergence_test`](@ref)
  (and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
  """
  function initial_condition_convergence_test(x, t, equations::CompressibleEulerMultichemistryEquations2D)
    c       = 2
    A       = 0.1
    L       = 2
    f       = 1/L
    omega   = 2 * pi * f
    ini     = c + A * sin(omega * (x[1] + x[2] - t))
  
    v1      = 1.0
    v2      = 1.0
  
    rho     = ini
  
    # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1)
    prim_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * rho for i in eachcomponent(equations))
  
    prim1 = rho * v1
    prim2 = rho * v2
    prim3 = rho^2
  
    prim_other = SVector{3, real(equations)}(prim1, prim2, prim3)
  
    return vcat(prim_other, prim_rho)
  end
  
  """
      source_terms_convergence_test(u, x, t, equations::CompressibleEulerMultichemistryEquations2D)
  Source terms used for convergence tests in combination with
  [`initial_condition_convergence_test`](@ref)
  (and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
  """
  @inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerMultichemistryEquations2D)
    # Same settings as in `initial_condition`
    c       = 2
    A       = 0.1
    L       = 2
    f       = 1/L
    omega   = 2 * pi * f
  
    gamma  = totalgamma(u, equations)
  
    x1, x2  = x
    si, co  = sincos((x1 + x2 - t)*omega)
    tmp1    = co * A * omega
    tmp2    = si * A
    tmp3    = gamma - 1
    tmp4    = (2*c - 1)*tmp3
    tmp5    = (2*tmp2*gamma - 2*tmp2 + tmp4 + 1)*tmp1
    tmp6    = tmp2 + c
  
    # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1
    du_rho  = SVector{ncomponents(equations), real(equations)}(2^(i-1) * (1-2)/(1-2^ncomponents(equations)) * tmp1 for i in eachcomponent(equations))
  
    du1 = tmp5
    du2 = tmp5
    du3 = 2*((tmp6 - 1.0)*tmp3 + tmp6*gamma)*tmp1
  
    du_other  = SVector{3, real(equations)}(du1, du2, du3)
  
    return vcat(du_other, du_rho)
  end
  

  function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, x, t,
                                        surface_flux_function, equations::CompressibleEulerMultichemistryEquations2D)

    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # rotate the internal solution state
    u_local = rotate_to_x(u_inner, normal, equations)

    # compute the primitive variables
    prim_local = cons2prim(u_local, equations)

    v_normal  = prim_local[1]
    v_tangent = prim_local[2]
    p_local   = prim_local[3]
    rho_local = sum(prim_local[4:end])

    gamma = totalgamma(u_local, equations) # or u_inner

    # Get the solution of the pressure Riemann problem
    # See Section 6.3.3 of
    # Eleuterio F. Toro (2009)
    # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
    # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
    if v_normal <= 0.0
      sound_speed = sqrt(gamma * p_local / rho_local) # local sound speed
      p_star = p_local * (1.0 + 0.5 * (gamma - 1) * v_normal / sound_speed)^(2.0 * gamma * (1 / (gamma - 1.0)))
    else # v_normal > 0.0
      A = 2.0 / ((gamma + 1) * rho_local)
      B = p_local * (gamma - 1) / (gamma + 1)
      p_star = p_local + 0.5 * v_normal / A * (v_normal + sqrt(v_normal^2 + 4.0 * A * (p_local + B)))
    end

    # For the slip wall we directly set the flux as the normal velocity is zero

    u_other = SVector{3, real(equations)}(p_star * normal[1] * norm_,
                                          p_star * normal[2] * norm_,
                                          zero(eltype(u_inner))) 
  
    u_rho = SVector{ncomponents(equations), real(equations)}(zero(eltype(u_inner)) for i in eachcomponent(equations))

    return vcat(u_other, u_rho)     

  end 

  function boundary_condition_slip_wall(u_inner, orientation, normal_direction, x, t,
                                        surface_flux_function, equations::CompressibleEulerMultichemistryEquations2D)

    if orientation == 1
      normal = SVector(1, 0)
    else 
      normal = SVector(0, 1)
    end

    return boundary_condition_slip_wall(u_inner, normal, x, t, surface_flux_function, equations)
  end


  function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, direction, x, t,
                                      surface_flux_function, equations::CompressibleEulerMultichemistryEquations2D)
    # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
    # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
    if isodd(direction)
      boundary_flux = -boundary_condition_slip_wall(u_inner, -normal_direction,
                                                    x, t, surface_flux_function, equations)
    else
      boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
                                                  x, t, surface_flux_function, equations)
    end

  return boundary_flux
  end

  function boundary_condition_dmr(u_inner, direction, x, t,
                                surface_flux_function,
                                equations::CompressibleEulerMultichemistryEquations2D)

  #u_boundary = prim2cons(SVector(rho, v1, v2, p), equations)
    if x[1] < 1/6

      rho_burnt = 8.0
      rho_unburnt = 0.0
      v1  = 8.25  * cosd(30)
      v2  = -8.25 * sind(30) 
      p   = 116.5
      u_boundary = prim2cons(SVector(v1, v2, p, rho_burnt, rho_unburnt), equations)

      # Calculate boundary flux
      flux = surface_flux_function(u_inner, u_boundary, direction, equations)

    else 

      u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3], u_inner[4], u_inner[5])
    #if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
      flux = surface_flux_function(u_inner, u_boundary, direction, equations)

    end

    return flux
  end


  @inline function chemistry_knallgas_5_detonation(u, dt, equations::CompressibleEulerMultichemistryEquations2D)
    # Same settings as in `initial_condition`
    v1, v2, p, H2, O2, OH, H2O, N2  = cons2prim(u, equations)
    @unpack heat_of_formations = equations     
    
    nmols = krome_nmols()[] # read Fortran module variable
    x = zeros(nmols) # default abundances (number density)
    
    idx_H2    = krome_idx_H2()[] + 1
    x[idx_H2] = H2/2.0
    idx_O2    = krome_idx_O2()[] + 1
    x[idx_O2] = O2/32.0
    idx_OH    = krome_idx_OH()[] + 1
    x[idx_OH] = OH/17.0
    idx_H2O   = krome_idx_H2O()[] + 1
    x[idx_H2O]= H2O/18.0
      
    rho   = H2 + O2 + OH + H2O + N2
    gamma = totalgamma(u, equations)
    
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
    
    Tgas  = fill(10.0 * (p/rho))
  
    dt = fill(dt) # time-step
    krome(x, Tgas, dt)
    
    x[idx_H2] = x[idx_H2]*2.0
    x[idx_O2] = x[idx_O2]*32.0
    x[idx_OH] = x[idx_OH]*17.0
    x[idx_H2O]= x[idx_H2O]*18.0
    
    chem = SVector{ncomponents(equations), real(equations)}(x[1], x[2], x[3], x[4], u[8])
    
    du_rho  = SVector{ncomponents(equations), real(equations)}(chem[i]-u[i+3] for i in eachcomponent(equations))
    
    du_other  = SVector{3, real(equations)}(0.0, 0.0, 0.0)
    
    return vcat(du_other, du_rho)
  end
  

  @inline function chemistry_knallgas_2_naive(u, dt, equations::CompressibleEulerMultichemistryEquations2D)
    # Same settings as in `initial_condition`
    v1, v2, p, burnt, unburnt = cons2prim(u, equations)
    @unpack heat_of_formations = equations     
  
    nmols = krome_nmols()[] # read Fortran module variable
    x = zeros(nmols) # default abundances (number density)
  
    idx_H2     = krome_idx_H2()[] + 1
    x[idx_H2]  = unburnt 
    idx_O2     = krome_idx_O2()[] + 1
    x[idx_O2]  = burnt
    
    rho   = burnt + unburnt 
    gamma = totalgamma(u, equations)
  
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
  
    Tgas = fill((p/rho))

    dt = fill(dt)

    krome(x, Tgas, dt)
  
    chem = SVector{ncomponents(equations), real(equations)}(x[2], x[1])
  
    du_rho  = SVector{ncomponents(equations), real(equations)}(chem[i]-u[i+3] for i in eachcomponent(equations))

    du_other  = SVector{3, real(equations)}(0.0, 0.0, 0.0)
  
    return vcat(du_other, du_rho)
  end

  
  """
      initial_condition_shock_bubble(x, t, equations::CompressibleEulerMultichemistryEquations2D{5, 2})
  A shock-bubble testcase for multichemistry Euler equations
  - Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
    Formulation of Entropy-Stable schemes for the multichemistry compressible Euler equations
    [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
  """
  function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMultichemistryEquations2D{5, 2})
    # bubble test case, see Gouasmi et al. https://arxiv.org/pdf/1904.00972
    # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
    # typical domain is rectangular, we change it to a square, as Trixi can only do squares
    @unpack gas_constants = equations
  
    # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
    delta   = 0.03
  
    # Region I
    rho1_1  = delta
    rho2_1  = 1.225 * gas_constants[1]/gas_constants[2] - delta
    v1_1    = zero(delta)
    v2_1    = zero(delta)
    p_1     = 101325
  
    # Region II
    rho1_2  = 1.225-delta
    rho2_2  = delta
    v1_2    = zero(delta)
    v2_2    = zero(delta)
    p_2     = 101325
  
    # Region III
    rho1_3  = 1.6861 - delta
    rho2_3  = delta
    v1_3    = -113.5243
    v2_3    = zero(delta)
    p_3     = 159060
  
    # Set up Region I & II:
    inicenter = SVector(zero(delta), zero(delta))
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
  
    if (x[1] > 0.50)
      # Set up Region III
      rho1    = rho1_3
      rho2    = rho2_3
      v1      = v1_3
      v2      = v2_3
      p       = p_3
    elseif (r < 0.25)
      # Set up Region I
      rho1    = rho1_1
      rho2    = rho2_1
      v1      = v1_1
      v2      = v2_1
      p       = p_1
    else
      # Set up Region II
      rho1    = rho1_2
      rho2    = rho2_2
      v1      = v1_2
      v2      = v2_2
      p       = p_2
    end
  
    return prim2cons(SVector(v1, v2, p, rho1, rho2), equations)
  end
  
  
  """
      initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMultichemistryEquations2D)
  A for multichemistry adapted weak blast wave taken from
  - Sebastian Hennemann, Gregor J. Gassner (2020)
    A provably entropy stable subcell shock capturing approach for high order split form DG
    [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
  """
  function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMultichemistryEquations2D)
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Set up polar coordinates
    inicenter         = SVector(0.0, 0.0)
    x_norm            = x[1] - inicenter[1]
    y_norm            = x[2] - inicenter[2]
    r                 = sqrt(x_norm^2 + y_norm^2)
    phi               = atan(y_norm, x_norm)
    sin_phi, cos_phi  = sincos(phi)
  
    prim_rho          = SVector{ncomponents(equations), real(equations)}(r > 0.5 ? 2^(i-1) * (1-2)/(1-2^ncomponents(equations))*1.0 : 2^(i-1) * (1-2)/(1-2^ncomponents(equations))*1.1691 for i in eachcomponent(equations))
  
    v1                = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2                = r > 0.5 ? 0.0 : 0.1882 * sin_phi
    p                 = r > 0.5 ? 1.0 : 1.245
  
    prim_other         = SVector{3, real(equations)}(v1, v2, p)
  
    return prim2cons(vcat(prim_other, prim_rho),equations)
  end
  
  
  # Calculate 1D flux for a single point
  @inline function flux(u, orientation::Integer, equations::CompressibleEulerMultichemistryEquations2D)
    rho_v1, rho_v2, rho_e  = u
    @unpack heat_of_formations = equations
  
    rho = density(u, equations)
  
    v1    = rho_v1/rho
    v2    = rho_v2/rho
    gamma = totalgamma(u, equations)
  
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
  
    p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2) - p_chem)     
  
    if orientation == 1
      f_rho = densities(u, v1, equations)
      f1  = rho_v1 * v1 + p
      f2  = rho_v2 * v1
      f3  = (rho_e + p) * v1
    else
      f_rho = densities(u, v2, equations)
      f1  = rho_v1 * v2
      f2  = rho_v2 * v2 + p
      f3  = (rho_e + p) * v2
    end

    f_other  = SVector{3, real(equations)}(f1, f2, f3)
  
    return vcat(f_other, f_rho)
  end
  
  
  # Calculate 1D flux for a single point in the normal direction
  # Note, this directional vector is not normalized
  @inline function flux(u, normal_direction::AbstractVector, equations::CompressibleEulerMultichemistryEquations2D)
    rho_v1, rho_v2, rho_e = u
    @unpack heat_of_formations = equations
  
    rho = density(u, equations)
  
    v1    = rho_v1/rho
    v2    = rho_v2/rho
    gamma = totalgamma(u, equations)
  
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
  
    p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2) - p_chem)     
  
    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    rho_v_normal = rho * v_normal
  
    f_rho = densities(u, v_normal, equations)
    f1 = rho_v_normal * v1 + p * normal_direction[1]
    f2 = rho_v_normal * v2 + p * normal_direction[2]
    f3 = (rho_e + p) * v_normal
  
    f_other  = SVector{3, real(equations)}(f1, f2, f3)
  
    return vcat(f_other, f_rho)
  end
  
  
  """
      flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMultichemistryEquations2D)
  Entropy conserving two-point flux by
  - Ayoub Gouasmi, Karthik Duraisamy (2020)
    "Formulation of Entropy-Stable schemes for the multichemistry compressible Euler equations""
    arXiv:1904.00972v3 [math.NA] 4 Feb 2020
  """
  @inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMultichemistryEquations2D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv, heat_of_formations = equations
    rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
    
    # for Chemistry
    rhok_mean_help = zeros(ncomponents(equations))
    for i in eachcomponent(equations)
      if u_ll[i+3] <= 0.0 || u_rr[i+3] <= 0.0# eps(Float32)
        rhok_mean_help[i] = 0.5* (u_ll[i+3] + u_rr[i+3])  # NEU jetzt mit Central!
      else
        rhok_mean_help[i] = ln_mean(u_ll[i+3], u_rr[i+3])
      end
    end
    #rhok_mean   = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i+3], u_rr[i+3]) for i in eachcomponent(equations))
    rhok_mean   = SVector{ncomponents(equations), real(equations)}(rhok_mean_help[i] for i in eachcomponent(equations))
    rhok_avg    = SVector{ncomponents(equations), real(equations)}(0.5 * (u_ll[i+3] + u_rr[i+3]) for i in eachcomponent(equations))

    # Iterating over all partial densities
    rho_ll      = density(u_ll, equations)
    rho_rr      = density(u_rr, equations)
  
    # extract velocities
    v1_ll       = rho_v1_ll/rho_ll
    v2_ll       = rho_v2_ll/rho_ll
    v1_rr       = rho_v1_rr/rho_rr
    v2_rr       = rho_v2_rr/rho_rr
    v1_avg      = 0.5 * (v1_ll + v1_rr)
    v2_avg      = 0.5 * (v2_ll + v2_rr)
    v1_square   = 0.5 * (v1_ll^2 + v1_rr^2)
    v2_square   = 0.5 * (v2_ll^2 + v2_rr^2)
    v_sum       = v1_avg + v2_avg
  
    enth      = zero(v_sum)
    help1_ll  = zero(v1_ll)
    help1_rr  = zero(v1_rr)
  
    for i in eachcomponent(equations)
      enth      += rhok_avg[i] * gas_constants[i]
      help1_ll  += u_ll[i+3] * cv[i]
      help1_rr  += u_rr[i+3] * cv[i]
    end

    # Chemistry
    p_chem_ll = zero(rho_ll)
    for i in eachcomponent(equations)                             
      p_chem_ll += heat_of_formations[i] * u_ll[i+3]   
    end
  
    p_chem_rr = zero(rho_rr)
    for i in eachcomponent(equations)                             
      p_chem_rr += heat_of_formations[i] * u_rr[i+3]                      
    end

    T_ll        = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2) - p_chem_ll) / help1_ll
    T_rr        = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2) - p_chem_rr) / help1_rr
    T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
    if T_ll <= 0.0 || T_rr <= 0.0
      println("T_ll: ",T_ll)
      println("T_rr: ",T_rr)
    end 
    T_log       = ln_mean(1.0/T_ll, 1.0/T_rr)
  
    # Calculate fluxes depending on orientation
    help1       = zero(T_ll)
    help2       = zero(T_rr)
    if orientation == 1
      f_rho       = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v1_avg for i in eachcomponent(equations))
      for i in eachcomponent(equations)
        help1     += f_rho[i] * cv[i]
        help2     += f_rho[i]
      end
      f1 = (help2) * v1_avg + enth/T
      f2 = (help2) * v2_avg
      f3 = (help1)/T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 + v2_avg * f2
    else
      f_rho       = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v2_avg for i in eachcomponent(equations))
      for i in eachcomponent(equations)
        help1     += f_rho[i] * cv[i]
        help2     += f_rho[i]
      end
      f1 = (help2) * v1_avg
      f2 = (help2) * v2_avg + enth/T
      f3 = (help1)/T_log - 0.5 * (v1_square + v2_square) * (help2) + v1_avg * f1 + v2_avg * f2
    end
    f_other  = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
  end
  

    """
      flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
                  equations::CompressibleEulerMulticomponentEquations2D)
  Adaption of the entropy conserving and kinetic energy preserving two-point flux by
  - Hendrik Ranocha (2018)
    Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
    for Hyperbolic Balance Laws
    [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
  See also
  - Hendrik Ranocha (2020)
    Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
    the Euler Equations Using Summation-by-Parts Operators
    [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
  """
  @inline function flux_ranocha(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMultichemistryEquations2D)
    # Unpack left and right state
    @unpack gammas, gas_constants, cv, heat_of_formations = equations
    rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

    # for Chemistry
    rhok_mean_help = zeros(ncomponents(equations))
    for i in eachcomponent(equations)
      if u_ll[i+3] <= 0.0 || u_rr[i+3] <= 0.0# eps(Float32)
        rhok_mean_help[i] = 0.5* (u_ll[i+3] + u_rr[i+3])  # NEU jetzt mit Central!
      else
        rhok_mean_help[i] = ln_mean(u_ll[i+3], u_rr[i+3])
      end
    end

    #rhok_mean   = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i+3], u_rr[i+3]) for i in eachcomponent(equations))
    rhok_mean   = SVector{ncomponents(equations), real(equations)}(rhok_mean_help[i] for i in eachcomponent(equations))
    rhok_avg    = SVector{ncomponents(equations), real(equations)}(0.5 * (u_ll[i+3] + u_rr[i+3]) for i in eachcomponent(equations))

    # Iterating over all partial densities
    rho_ll      = density(u_ll, equations)
    rho_rr      = density(u_rr, equations)

    # Calculating gamma
    gamma               = totalgamma(0.5*(u_ll+u_rr), equations)
    inv_gamma_minus_one = 1/(gamma-1) 

    # Chemistry
    p_chem_ll = zero(rho_ll)
    for i in eachcomponent(equations)                             
      p_chem_ll += heat_of_formations[i] * u_ll[i+3]                      
    end
  
    p_chem_rr = zero(rho_rr)
    for i in eachcomponent(equations)                             
      p_chem_rr += heat_of_formations[i] * u_rr[i+3]                      
    end

    # extract velocities
    v1_ll               = rho_v1_ll / rho_ll
    v1_rr               = rho_v1_rr / rho_rr
    v1_avg              = 0.5 * (v1_ll + v1_rr)
    v2_ll               = rho_v2_ll / rho_ll 
    v2_rr               = rho_v2_rr / rho_rr
    v2_avg              = 0.5 * (v2_ll + v2_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr)

    # helpful variables
    help1_ll  = zero(v1_ll)
    help1_rr  = zero(v1_rr)
    enth_ll   = zero(v1_ll)
    enth_rr   = zero(v1_rr)
    for i in eachcomponent(equations)
      enth_ll   += u_ll[i+3] * gas_constants[i]
      enth_rr   += u_rr[i+3] * gas_constants[i]
      help1_ll  += u_ll[i+3] * cv[i]
      help1_rr  += u_rr[i+3] * cv[i]
    end

    # temperature and pressure
    T_ll            = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2) - p_chem_ll) / help1_ll
    T_rr            = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2) - p_chem_rr) / help1_rr
    p_ll            = T_ll * enth_ll 
    p_rr            = T_rr * enth_rr
    p_avg           = 0.5 * (p_ll + p_rr)
    if rho_ll <= 0 || rho_rr <= 0 || p_ll <= 0 || p_rr <= 0
      println("rho_ll: ",rho_ll)
      println("rho_rr: ",rho_rr)
      println("p_ll: ",p_ll)
      println("p_rr: ",p_rr)
    end 
    inv_rho_p_mean  = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)

    f_rho_sum = zero(T_rr)
    if orientation == 1
      f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v1_avg for i in eachcomponent(equations))
      for i in eachcomponent(equations)
        f_rho_sum += f_rho[i]
      end
      f1 = f_rho_sum * v1_avg + p_avg
      f2 = f_rho_sum * v2_avg
      f3 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) + 0.5 * (p_ll*v1_rr + p_rr*v1_ll)
    else
      f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i]*v2_avg for i in eachcomponent(equations))
      for i in eachcomponent(equations)
        f_rho_sum += f_rho[i]
      end
      f1 = f_rho_sum * v1_avg
      f2 = f_rho_sum * v2_avg + p_avg
      f3 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) + 0.5 * (p_ll*v2_rr + p_rr*v2_ll)
    end

    # momentum and energy flux
    f_other  = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
  end


  function flux_hllc(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMultichemistryEquations2D)
    # Calculate primitive variables and speed of sound
    @unpack heat_of_formations = equations
  
    rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
    rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)
  
    gamma_ll = totalgamma(u_ll, equations)
    gamma_rr = totalgamma(u_rr, equations)
    gamma = 0.5 * (gamma_ll + gamma_rr)

    p_chem_ll = zero(rho_ll)
    for i in eachcomponent(equations)                             
      p_chem_ll += heat_of_formations[i] * u_ll[i+3]                      
    end
  
    p_chem_rr = zero(rho_rr)
    for i in eachcomponent(equations)                             
      p_chem_rr += heat_of_formations[i] * u_rr[i+3]                      
    end
  
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    e_ll  = rho_e_ll / rho_ll
    p_ll = (gamma_ll - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2) - p_chem_ll)
    if p_ll < 0 || rho_ll < 0
      println("p_rr: ",p_ll) 
      println("rho_rr: ",rho_ll)
    end
    c_ll = sqrt(gamma_ll*p_ll/rho_ll)
  
    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    e_rr  = rho_e_rr / rho_rr
    p_rr = (gamma_rr - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2) - p_chem_rr)
    if p_rr < 0 || rho_rr < 0
      println("p_rr: ",p_rr) 
      println("rho_rr: ",rho_rr)
    end
    c_rr = sqrt(gamma_rr*p_rr/rho_rr)
  
    # Obtain left and right fluxes
    f_ll = flux(u_ll, orientation, equations)
    f_rr = flux(u_rr, orientation, equations)
  
    # Compute Roe averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
  
    if orientation == 1 # x-direction
      vel_L = v1_ll
      vel_R = v1_rr
      ekin_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2 
    elseif orientation == 2 # y-direction
      vel_L = v2_ll
      vel_R = v2_rr
      ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2 
    end
  
    vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
    ekin_roe = 0.5 * (vel_roe^2 + ekin_roe / sum_sqrt_rho^2)
    H_ll = (rho_e_ll + p_ll) / rho_ll
    H_rr = (rho_e_rr + p_rr) / rho_rr 

    H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho

    if (H_roe-ekin_roe) < 0
      println("H_roe: ",H_roe)
      println("ekin_roe: ",ekin_roe)
      println("H_roe - ekin_roe: ",(H_roe - ekin_roe))
    end 
    c_roe = sqrt((gamma - 1) * (H_roe - ekin_roe))
    Ssl = min(vel_L - c_ll, vel_roe - c_roe)
    Ssr = max(vel_R + c_rr, vel_roe + c_roe)
    sMu_L = Ssl - vel_L
    sMu_R = Ssr - vel_R
  
    if Ssl >= 0.0 && Ssr > 0.0
      f_rho = SVector{ncomponents(equations), real(equations)}(f_ll[i+3] for i in eachcomponent(equations))
      f_other = SVector{3, real(equations)}(f_ll[1], f_ll[2], f_ll[3])
    elseif Ssr <= 0.0 && Ssl < 0.0
      f_rho = SVector{ncomponents(equations), real(equations)}(f_rr[i+3] for i in eachcomponent(equations))
      f_other = SVector{3, real(equations)}(f_rr[1], f_rr[2], f_rr[3])
    else
      SStar = (p_rr - p_ll + rho_ll*vel_L*sMu_L - rho_rr*vel_R*sMu_R) / (rho_ll*sMu_L - rho_rr*sMu_R)
      if Ssl <= 0.0 <= SStar
        densStar = rho_ll*sMu_L / (Ssl-SStar)
        enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
        UStar1 = densStar
        UStar4 = densStar*enerStar
        if orientation == 1 # x-direction
          UStar2 = densStar*SStar
          UStar3 = densStar*v2_ll
        elseif orientation == 2 # y-direction
          UStar2 = densStar*v1_ll
          UStar3 = densStar*SStar
        end
        f2 = f_ll[1]+Ssl*(UStar2 - rho_v1_ll)
        f3 = f_ll[2]+Ssl*(UStar3 - rho_v2_ll)
        f4 = f_ll[3]+Ssl*(UStar4 - rho_e_ll)
        f_rho = SVector{ncomponents(equations), real(equations)}(f_ll[i+3]+Ssl*((u_ll[i+3]*sMu_L/(Ssl-SStar)) - u_ll[i+3]) for i in eachcomponent(equations))
        f_other = SVector{3, real(equations)}(f2, f3, f4)
      else
        densStar = rho_rr*sMu_R / (Ssr-SStar)
        enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
        UStar1 = densStar
        UStar4 = densStar*enerStar
        if orientation == 1 # x-direction
          UStar2 = densStar*SStar
          UStar3 = densStar*v2_rr
        elseif orientation == 2 # y-direction
          UStar2 = densStar*v1_rr
          UStar3 = densStar*SStar
        end
        f2 = f_rr[1]+Ssr*(UStar2 - rho_v1_rr)
        f3 = f_rr[2]+Ssr*(UStar3 - rho_v2_rr)
        f4 = f_rr[3]+Ssr*(UStar4 - rho_e_rr)
        f_rho = SVector{ncomponents(equations), real(equations)}(f_rr[i+3]+Ssr*((u_rr[i+3]*sMu_R/(Ssr-SStar)) - u_rr[i+3]) for i in eachcomponent(equations))
        f_other = SVector{3, real(equations)}(f2, f3, f4)
      end
    end
    return vcat(f_other, f_rho)
  end


  # function flux_hllc_new(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMultichemistryEquations2D)
  #   # Calculate primitive variables and speed of sound
  #   @unpack heat_of_formations = equations
  
  #   v1_ll, v2_ll, p_ll = cons2prim(u_ll)
  #   v1_rr, v2_rr, p_rr = cons2prim(u_rr)

  #   rho_e_ll = u_ll[3]
  #   rho_e_rr = u_rr[3]
  
  #   rho_ll = density(u_ll, equations)
  #   rho_rr = density(u_rr, equations)

  #   gamma_ll = totalgamma(u_ll, equations)
  #   gamma_rr = totalgamma(u_rr, equations)
  #   gamma = 0.5 * (gamma_ll + gamma_rr)

  #   beta = sqrt(0.5*(gamma-1)/gamma)

  #   p_chem_ll = zero(rho_ll)
  #   for i in eachcomponent(equations)                             
  #     p_chem_ll += heat_of_formations[i] * u_ll[i+3]                      
  #   end
  
  #   p_chem_rr = zero(rho_rr)
  #   for i in eachcomponent(equations)                             
  #     p_chem_rr += heat_of_formations[i] * u_rr[i+3]                      
  #   end

  #   # Sound Speed
  #   c_ll = sqrt(gamma*p_ll/rho_ll)
  #   c_rr = sqrt(gamma*p_rr/rho_rr)

  #   # Enthalpy
  #   H_ll = rho_e_ll + p_ll/rho_ll 
  #   H_rr = rho_e_rr + p_rr/rho_rr

  #   if orientation == 1 # x-direction
  #     v_ll = v1_ll
  #     v_rr = v1_rr
  #   elseif orientation == 2 # y-direction
  #     v_ll = v2_ll
  #     v_rr = v2_rr
  #   end

  #   # Roe Averages
  #   hTilde = (sqrt(rho_ll)*H_ll) + sqrt(rho_rr)*H_rr) / (sqrt(rho_ll) + sqrt(rho_rr))
  #   vTilde = (sqrt(rho_ll)*v_ll) + sqrt(rho_rr)*v_rr) / (sqrt(rho_ll) + sqrt(rho_rr))
  #   aTilde = sqrt((gamma-1.0)*(hTilde - 0.5 * vTilde^2))

  #   # Signal velocities
  #   S_ll = min(vTilde - aTilde, v1_ll - beta * c_ll, 0.0)
  #   S_rr = max(vTilde + aTilde, v1_rr - beta * c_rr, 0.0)

  #   # Obtain left and right fluxes
  #   f_ll = flux(u_ll, orientation, equations)
  #   f_rr = flux(u_rr, orientation, equations)

  #   if S_ll >= 0.0 && S_rr > 0.0
  #     f_rho = SVector{ncomponents(equations), real(equations)}(f_ll[i+3] for i in eachcomponent(equations))
  #     f_other = SVector{3, real(equations)}(f_ll[1], f_ll[2], f_ll[3])
  #   elseif S_rr <= 0.0 && S_ll < 0.0
  #     f_rho = SVector{ncomponents(equations), real(equations)}(f_rr[i+3] for i in eachcomponent(equations))
  #     f_other = SVector{3, real(equations)}(f_rr[1], f_rr[2], f_rr[3]) 
  #   else
  #     SStar = (p_rr - p_ll + rho_ll*v_ll*(S_ll - v_ll) - rho_rr*v_rr*(S_rr - v_rr)) / (rho_ll*(S_ll - v_ll) - rho_rr*(S_rr - v_rr))
  #     UStar = 0.0

  #     if S_ll <= 0.0 && 0.0 <= SStar
  #       QStar = rho_ll*(S_ll - v_ll)/(S_ll-SStar)

  #       UStar1 = QStar
  #       UStar2 = QStar * SStar
  #       UStar3 = QStar * v2_ll 

  # !!!!!  BIS HIERHIN NEU !!!!!

  #       UStar4 = QStar * (U_L(ENER_PRIM) + (SStar-V_L)*(SStar + p_L/U_L(DENS_PRIM)/(S_L - V_L)))

  #   # Compute Roe averages
  #   sqrt_rho_ll = sqrt(rho_ll)
  #   sqrt_rho_rr = sqrt(rho_rr)
  #   sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
  
  #   if orientation == 1 # x-direction
  #     vel_L = v1_ll
  #     vel_R = v1_rr
  #     ekin_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2 
  #   elseif orientation == 2 # y-direction
  #     vel_L = v2_ll
  #     vel_R = v2_rr
  #     ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2 
  #   end
  
  #   vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
  #   ekin_roe = 0.5 * (vel_roe^2 + ekin_roe / sum_sqrt_rho^2)
 

  #   H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho

  #   if (H_roe-ekin_roe) < 0
  #     println("H_roe: ",H_roe)
  #     println("ekin_roe: ",ekin_roe)
  #     println("H_roe - ekin_roe: ",(H_roe - ekin_roe))
  #   end 
  #   c_roe = sqrt((gamma - 1) * (H_roe - ekin_roe))
  #   Ssl = min(vel_L - c_ll, vel_roe - c_roe)
  #   Ssr = max(vel_R + c_rr, vel_roe + c_roe)
  #   sMu_L = Ssl - vel_L
  #   sMu_R = Ssr - vel_R
  
  #   if Ssl >= 0.0 && Ssr > 0.0
  #     f_rho = SVector{ncomponents(equations), real(equations)}(f_ll[i+3] for i in eachcomponent(equations))
  #     f_other = SVector{3, real(equations)}(f_ll[1], f_ll[2], f_ll[3])
  #   elseif Ssr <= 0.0 && Ssl < 0.0
  #     f_rho = SVector{ncomponents(equations), real(equations)}(f_rr[i+3] for i in eachcomponent(equations))
  #     f_other = SVector{3, real(equations)}(f_rr[1], f_rr[2], f_rr[3])
  #   else
  #     SStar = (p_rr - p_ll + rho_ll*vel_L*sMu_L - rho_rr*vel_R*sMu_R) / (rho_ll*sMu_L - rho_rr*sMu_R)
  #     if Ssl <= 0.0 <= SStar
  #       densStar = rho_ll*sMu_L / (Ssl-SStar)
  #       enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
  #       UStar1 = densStar
  #       UStar4 = densStar*enerStar
  #       if orientation == 1 # x-direction
  #         UStar2 = densStar*SStar
  #         UStar3 = densStar*v2_ll
  #       elseif orientation == 2 # y-direction
  #         UStar2 = densStar*v1_ll
  #         UStar3 = densStar*SStar
  #       end
  #       f2 = f_ll[1]+Ssl*(UStar2 - rho_v1_ll)
  #       f3 = f_ll[2]+Ssl*(UStar3 - rho_v2_ll)
  #       f4 = f_ll[3]+Ssl*(UStar4 - rho_e_ll)
  #       f_rho = SVector{ncomponents(equations), real(equations)}(f_ll[i+3]+Ssl*((u_ll[i+3]*sMu_L/(Ssl-SStar)) - u_ll[i+3]) for i in eachcomponent(equations))
  #       f_other = SVector{3, real(equations)}(f2, f3, f4)
  #     else
  #       densStar = rho_rr*sMu_R / (Ssr-SStar)
  #       enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
  #       UStar1 = densStar
  #       UStar4 = densStar*enerStar
  #       if orientation == 1 # x-direction
  #         UStar2 = densStar*SStar
  #         UStar3 = densStar*v2_rr
  #       elseif orientation == 2 # y-direction
  #         UStar2 = densStar*v1_rr
  #         UStar3 = densStar*SStar
  #       end
  #       f2 = f_rr[1]+Ssr*(UStar2 - rho_v1_rr)
  #       f3 = f_rr[2]+Ssr*(UStar3 - rho_v2_rr)
  #       f4 = f_rr[3]+Ssr*(UStar4 - rho_e_rr)
  #       f_rho = SVector{ncomponents(equations), real(equations)}(f_rr[i+3]+Ssr*((u_rr[i+3]*sMu_R/(Ssr-SStar)) - u_rr[i+3]) for i in eachcomponent(equations))
  #       f_other = SVector{3, real(equations)}(f2, f3, f4)
  #     end
  #   end
  #   return vcat(f_other, f_rho)
  # end
  
  
  @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMultichemistryEquations2D)
    v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
  
    # Get the velocity value in the appropriate direction
    if orientation == 1
      v_ll = v1_ll
      v_rr = v1_rr
    else # orientation == 2
      v_ll = v2_ll
      v_rr = v2_rr
    end
    # Calculate sound speeds
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    gamma_ll = totalgamma(u_ll, equations)
    gamma_rr = totalgamma(u_rr, equations)

    c_ll = sqrt(gamma_ll * p_ll / rho_ll)
    c_rr = sqrt(gamma_rr * p_rr / rho_rr)
  
    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
  end
  
  
  @inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleEulerMultichemistryEquations2D)
    v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
  
    # Calculate normal velocities and sound speed
    # left
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    gamma_ll = totalgamma(u_ll, equations)
    gamma_rr = totalgamma(u_rr, equations)


    v_ll = (  v1_ll * normal_direction[1]
            + v2_ll * normal_direction[2] )
    c_ll = sqrt(gamma_ll * p_ll / rho_ll)
    # right
    v_rr = (  v1_rr * normal_direction[1]
            + v2_rr * normal_direction[2] )
    c_rr = sqrt(gamma_rr * p_rr / rho_rr)
  
    return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
  end
 
  
  @inline function max_abs_speeds(u, equations::CompressibleEulerMultichemistryEquations2D)
    rho_v1, rho_v2, rho_e = u
    @unpack heat_of_formations = equations
  
    rho   = density(u, equations)
    v1    = rho_v1 / rho
    v2    = rho_v2 / rho
  
    gamma = totalgamma(u, equations)
  
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
  
    p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2) - p_chem)    
  
    #p     = (gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
    if p < 0 || rho < 0
      println("p: ", p)
      println("rho: ",rho)
    end 
    c     = sqrt(gamma * p / rho)
  
    return (abs(v1) + c, abs(v2) + c, )
  end
  

  # Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
  # has been normalized prior to this rotation of the state vector
  @inline function rotate_to_x(u, normal_vector, equations::CompressibleEulerMultichemistryEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D rotation matrix with normal and tangent directions of the form
    # [ 1    0    0   0;
    #   0   n_1  n_2  0;
    #   0   t_1  t_2  0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    u_other = SVector(c * u[1] + s * u[2],
                      -s * u[1] + c * u[2],
                      u[3])

    u_rho = SVector{ncomponents(equations), real(equations)}(u[3+i] for i in eachcomponent(equations))

    return vcat(u_other, u_rho)
  end


  # Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
  # has been normalized prior to this back-rotation of the state vector
  @inline function rotate_from_x(u, normal_vector, equations::CompressibleEulerMultichemistryEquations2D)
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D back-rotation matrix with normal and tangent directions of the form
    # [ 1    0    0   0;
    #   0   n_1  t_1  0;
    #   0   n_2  t_2  0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    u_other = SVector(c * u[1] - s * u[2],
                      s * u[1] + c * u[2],
                      u[3])

    u_rho = SVector{ncomponents(equations), real(equations)}(u[3+i] for i in eachcomponent(equations))


    return vcat(u_other, u_rho)
  end

  
  # Convert conservative variables to primitive
  @inline function cons2prim(u, equations::CompressibleEulerMultichemistryEquations2D)
    rho_v1, rho_v2, rho_e = u
    @unpack heat_of_formations = equations
  
    prim_rho = SVector{ncomponents(equations), real(equations)}(u[i+3] for i in eachcomponent(equations))
  
    rho   = density(u, equations)
    v1    = rho_v1 / rho
    v2    = rho_v2 / rho
    gamma = totalgamma(u, equations)
  
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
  
    p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2) - p_chem)    
  
    prim_other =  SVector{3, real(equations)}(v1, v2, p)
  
    return vcat(prim_other, prim_rho)
  end
  
  
  # Convert conservative variables to entropy
  @inline function cons2entropy(u, equations::CompressibleEulerMultichemistryEquations2D)
    @unpack cv, gammas, gas_constants, heat_of_formations = equations
    rho_v1, rho_v2, rho_e = u
  
    rho       = density(u, equations)
  
    v1        = rho_v1 / rho
    v2        = rho_v2 / rho
    v_square  = v1^2 + v2^2
    gamma     = totalgamma(u, equations)
    
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
  
    p     = (gamma - 1) * (rho_e - 0.5 * rho * (v_square) - p_chem)    
    #p         = (gamma - 1) * (rho_e - 0.5 * rho * v_square)
    s         = log(p) - gamma*log(rho)
    rho_p     = rho / p
  
    # Multichemistry stuff
    help1 = zero(v1)
  
    for i in eachcomponent(equations)
      help1 += u[i+3] * cv[i]
    end
  
    T         = (rho_e - 0.5 * rho * v_square - p_chem) / (help1)
  
    #entrop_rho  = SVector{ncomponents(equations), real(equations)}(1.0 * (cv[i] * log(T) - gas_constants[i] * log(u[i+3])) + gas_constants[i] + cv[i] - (v_square / (2*T)) for i in eachcomponent(equations))
    entrop_rho  = SVector{ncomponents(equations), real(equations)}(i for i in eachcomponent(equations)) #-1.0 * (cv[i] * log(T) - gas_constants[i] * log(u[i+3])) + gas_constants[i] + cv[i] - (v_square / (2*T)) for i in eachcomponent(equations))


    w1        = v1/T
    w2        = v2/T
    w3        = -1.0/T
  
    entrop_other = SVector{3, real(equations)}(w1, w2, w3)
  
    return vcat(entrop_other, entrop_rho)
  end
  
  
  # Convert primitive to conservative variables
  @inline function prim2cons(prim, equations::CompressibleEulerMultichemistryEquations2D)
    @unpack cv, gammas, heat_of_formations = equations
    v1, v2, p = prim
  
    cons_rho = SVector{ncomponents(equations), real(equations)}(prim[i+3] for i in eachcomponent(equations))
    rho     = density(prim, equations)
    gamma   = totalgamma(prim, equations)
  
    rho_v1  = rho * v1
    rho_v2  = rho * v2
    
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * prim[i+3]                      
    end
  
    #p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2) - p_chem)    
    rho_e   = p/(gamma-1) + 0.5 * (rho * v1^2 + rho * v2^2) + p_chem
  
    cons_other = SVector{3, real(equations)}(rho_v1, rho_v2, rho_e)
  
    return vcat(cons_other, cons_rho)
  end
  
  
  """
      totalgamma(u, equations::CompressibleEulerMultichemistryEquations2D)
  Function that calculates the total gamma out of all partial gammas using the
  partial density fractions as well as the partial specific heats at constant volume.
  """
  @inline function totalgamma(u, equations::CompressibleEulerMultichemistryEquations2D)
    @unpack cv, gammas = equations
  
    help1 = zero(u[1])
    help2 = zero(u[1])
  
    for i in eachcomponent(equations)
      help1 += u[i+3] * cv[i] * gammas[i]
      help2 += u[i+3] * cv[i]
    end
  
    return help1/help2
  end
  
  
  @inline function density_pressure(u, equations::CompressibleEulerMultichemistryEquations2D)
    rho_v1, rho_v2, rho_e = u
    @unpack heat_of_formations = equations
  
    rho          = density(u, equations)
    gamma        = totalgamma(u, equations)
  
    v1  = rho_v1 / rho
    v2  = rho_v2 / rho
  
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
  
    p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2) - p_chem)    
  
    #rho_times_p  = (gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2) - p_chem)
  
    return p * rho
  end
  
  
  @inline function pressure(u, equations::CompressibleEulerMultichemistryEquations2D)
    rho_v1, rho_v2, rho_e = u
    @unpack heat_of_formations = equations
  
    rho          = density(u, equations)
    gamma        = totalgamma(u, equations)
  
    v1  = rho_v1 / rho
    v2  = rho_v2 / rho
  
    p_chem = zero(rho)
    for i in eachcomponent(equations)                             
      p_chem += heat_of_formations[i] * u[i+3]                      
    end
  
    p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2) - p_chem)    
  
    #rho_times_p  = (gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))
  
    return p
  end


  @inline function dpdu(u, equations::CompressibleEulerMultichemistryEquations2D)
    rho = density(u, equations)
  
    gamma = totalgamma(u, equations)
  
    dpdu1 = (gamma-1) * (-u[1])/rho 
    dpdu2 = (gamma-1) * (-u[2])/rho 
    dpdu3 = (gamma-1)

    dpdu_other = SVector{3, real(equations)}(dpdu1, dpdu2, dpdu3) 

    dpdu_rho = SVector{ncomponents(equations), real(equations)}((gamma-1) * 0.5 * ((u[1]/rho)^2 + (u[2]/rho)^2) for i in eachcomponent(equations))
  
    return vcat(dpdu_other, dpdu_rho)
   end
  
  
  @inline function density(u, equations::CompressibleEulerMultichemistryEquations2D)
    rho = zero(u[1])
  
    for i in eachcomponent(equations)
      rho += u[i+3]
    end
  
    return rho
   end
  
   @inline function densities(u, v, equations::CompressibleEulerMultichemistryEquations2D)
  
    return SVector{ncomponents(equations), real(equations)}(u[i+3]*v for i in eachcomponent(equations))
   end
  
  
  end # @muladd