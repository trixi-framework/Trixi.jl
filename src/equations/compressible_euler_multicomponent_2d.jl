# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    CompressibleEulerMulticomponentEquations2D(; gammas, gas_constants)

Multicomponent version of the compressible Euler equations
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
struct CompressibleEulerMulticomponentEquations2D{NVARS, NCOMP, RealT<:Real} <: AbstractCompressibleEulerMulticomponentEquations{2, NVARS, NCOMP}
  gammas            ::SVector{NCOMP, RealT}
  gas_constants     ::SVector{NCOMP, RealT}
  cv                ::SVector{NCOMP, RealT}
  cp                ::SVector{NCOMP, RealT}

  function CompressibleEulerMulticomponentEquations2D{NVARS, NCOMP, RealT}(gammas       ::SVector{NCOMP, RealT},
                                                                           gas_constants::SVector{NCOMP, RealT}) where {NVARS, NCOMP, RealT<:Real}

    NCOMP >= 1 || throw(DimensionMismatch("`gammas` and `gas_constants` have to be filled with at least one value"))

    cv = gas_constants ./ (gammas .- 1)
    cp = gas_constants + gas_constants ./ (gammas .- 1)

    new(gammas, gas_constants,cv, cp)
  end
end


function CompressibleEulerMulticomponentEquations2D(; gammas, gas_constants)

  _gammas        = promote(gammas...)
  _gas_constants = promote(gas_constants...)
  RealT          = promote_type(eltype(_gammas), eltype(_gas_constants))

  NVARS = length(_gammas) + 3
  NCOMP = length(_gammas)

  __gammas        = SVector(map(RealT, _gammas))
  __gas_constants = SVector(map(RealT, _gas_constants))

  return CompressibleEulerMulticomponentEquations2D{NVARS, NCOMP, RealT}(__gammas, __gas_constants)
end


@inline Base.real(::CompressibleEulerMulticomponentEquations2D{NVARS, NCOMP, RealT}) where {NVARS, NCOMP, RealT} = RealT


function varnames(::typeof(cons2cons), equations::CompressibleEulerMulticomponentEquations2D)

  cons  = ("rho_v1", "rho_v2", "rho_e")
  rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
  return (cons..., rhos...)
end


function varnames(::typeof(cons2prim), equations::CompressibleEulerMulticomponentEquations2D)

  prim  = ("v1", "v2", "p")
  rhos  = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
  return (prim..., rhos...)
end


# Set initial conditions at physical location `x` for time `t`

"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerMulticomponentEquations2D)
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
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerMulticomponentEquations2D)
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


"""
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{5, 2})

A shock-bubble testcase for multicomponent Euler equations
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{5, 2})
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
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{6, 3})

Adaption of the shock-bubble testcase for multicomponent Euler equations to 3 components
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{6, 3})
  # bubble test case, see Gouasmi et al. https://arxiv.org/pdf/1904.00972
  # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
  # adapted to 3 component testcase
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares

  @unpack gas_constants = equations

  # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
  delta   = 0.05

  # Region Ia
  rho1_1a   = delta
  rho2_1a   = 1.225 * gas_constants[1]/gas_constants[2] - delta
  rho3_1a   = delta
  v1_1a     = zero(delta)
  v2_1a     = zero(delta)
  p_1a      = 101325

  # Region Ib
  rho1_1b   = delta
  rho2_1b   = delta
  rho3_1b   = 1.225 * gas_constants[1]/gas_constants[3] - delta
  v1_1b     = zero(delta)
  v2_1b     = zero(delta)
  p_1b      = 101325

  # Region II
  rho1_2    = 1.225-delta
  rho2_2    = delta
  rho3_2    = delta
  v1_2      = zero(delta)
  v2_2      = zero(delta)
  p_2       = 101325

  # Region III
  rho1_3    = 1.6861 - delta
  rho2_3    = delta
  rho3_3    = delta
  v1_3      = -113.5243
  v2_3      = zero(delta)
  p_3       = 159060

  # Set up Region I & II:
  inicenter_a = SVector(zero(delta), 1.8)
  x_norm_a    = x[1] - inicenter_a[1]
  y_norm_a    = x[2] - inicenter_a[2]
  r_a         = sqrt(x_norm_a^2 + y_norm_a^2)

  inicenter_b = SVector(zero(delta), -1.8)
  x_norm_b    = x[1] - inicenter_b[1]
  y_norm_b    = x[2] - inicenter_b[2]
  r_b         = sqrt(x_norm_b^2 + y_norm_b^2)


  if (x[1] > 0.5)
    # Set up Region III
    rho1    = rho1_3
    rho2    = rho2_3
    rho3    = rho3_3
    v1      = v1_3
    v2      = v2_3
    p       = p_3
  elseif (r_a < 0.25)
    # Set up Region I
    rho1    = rho1_1a
    rho2    = rho2_1a
    rho3    = rho3_1a
    v1      = v1_1a
    v2      = v2_1a
    p       = p_1a
  elseif (r_b < 0.25)
    rho1    = rho1_1b
    rho2    = rho2_1b
    rho3    = rho3_1b
    v1      = v1_1b
    v2      = v2_1b
    p       = p_1b
  else
    # Set up Region II
    rho1    = rho1_2
    rho2    = rho2_2
    rho3    = rho3_2
    v1      = v1_2
    v2      = v2_2
    p       = p_2
  end

  return prim2cons(SVector(v1, v2, p, rho1, rho2, rho3), equations)
end



"""
    initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{10, 7})

Adaption of the shock-bubble testcase for multicomponent Euler equations to 7 components
- Ayoub Gouasmi, Karthik Duraisamy, Scott Murman
  Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations
  [arXiv: 1904.00972](https://arxiv.org/abs/1904.00972)
"""
function initial_condition_shock_bubble(x, t, equations::CompressibleEulerMulticomponentEquations2D{10, 7})
  # bubble test case, see Gouasmi et al. https://arxiv.org/pdf/1904.00972
  # other reference: https://www.researchgate.net/profile/Pep_Mulet/publication/222675930_A_flux-split_algorithm_applied_to_conservative_models_for_multicomponent_compressible_flows/links/568da54508aeaa1481ae7af0.pdf
  # adapted to 7 component testcase
  # typical domain is rectangular, we change it to a square, as Trixi can only do squares

  @unpack gas_constants = equations

  # Positivity Preserving Parameter, can be set to zero if scheme is positivity preserving
  delta   = 0.05

  # Region Ia
  rho1_1a   = delta
  rho2_1a   = 1.274 * gas_constants[1]/gas_constants[2] - delta
  rho3_1a   = delta
  rho4_1a   = delta
  rho5_1a   = delta
  rho6_1a   = delta
  rho7_1a   = delta
  v1_1a     = zero(delta)
  v2_1a     = zero(delta)
  p_1a      = 101325

  # Region Ib
  rho1_1b   = delta
  rho2_1b   = delta
  rho3_1b   = 1.274 * gas_constants[1]/gas_constants[3] - delta
  rho4_1b   = delta
  rho5_1b   = delta
  rho6_1b   = delta
  rho7_1b   = delta
  v1_1b     = zero(delta)
  v2_1b     = zero(delta)
  p_1b      = 101325

  # Region Ic
  rho1_1c   = delta
  rho2_1c   = delta
  rho3_1c   = delta
  rho4_1c   = 1.274 * gas_constants[1]/gas_constants[4] - delta
  rho5_1c   = delta
  rho6_1c   = delta
  rho7_1c   = delta
  v1_1c     = zero(delta)
  v2_1c     = zero(delta)
  p_1c      = 101325

  # Region Id
  rho1_1d   = delta
  rho2_1d   = delta
  rho3_1d   = delta
  rho4_1d   = delta
  rho5_1d   = 1.274 * gas_constants[1]/gas_constants[5] - delta
  rho6_1d   = delta
  rho7_1d   = delta
  v1_1d     = zero(delta)
  v2_1d     = zero(delta)
  p_1d      = 101325

  # Region Ie
  rho1_1e   = delta
  rho2_1e   = delta
  rho3_1e   = delta
  rho4_1e   = delta
  rho5_1e   = delta
  rho6_1e   = 1.274 * gas_constants[1]/gas_constants[6] - delta
  rho7_1e   = delta
  v1_1e     = zero(delta)
  v2_1e     = zero(delta)
  p_1e      = 101325

  # Region If
  rho1_1f   = delta
  rho2_1f   = delta
  rho3_1f   = delta
  rho4_1f   = delta
  rho5_1f   = delta
  rho6_1f   = delta
  rho7_1f   = 1.274 * gas_constants[1]/gas_constants[7] - delta
  v1_1f     = zero(delta)
  v2_1f     = zero(delta)
  p_1f      = 101325

  # Region II
  rho1_2    = 1.225-delta
  rho2_2    = delta
  rho3_2    = delta
  rho4_2    = delta
  rho5_2    = delta
  rho6_2    = delta
  rho7_2    = delta
  v1_2      = zero(delta)
  v2_2      = zero(delta)
  p_2       = 101325

  # Region III
  rho1_3    = 1.6861 - delta
  rho2_3    = delta
  rho3_3    = delta
  rho4_3    = delta
  rho5_3    = delta
  rho6_3    = delta
  rho7_3    = delta
  v1_3      = -113.5243
  v2_3      = zero(delta)
  p_3       = 159060

  # Set up Region I & II:
  #inicenter_a = SVector(zero(delta), 1.8)
  inicenter_a = SVector(zero(delta), 2.55)
  x_norm_a    = x[1] - inicenter_a[1]
  y_norm_a    = x[2] - inicenter_a[2]
  r_a         = sqrt(x_norm_a^2 + y_norm_a^2)

  #inicenter_b = SVector(zero(delta), -1.8)
  inicenter_b = SVector(zero(delta), 1.60)
  x_norm_b    = x[1] - inicenter_b[1]
  y_norm_b    = x[2] - inicenter_b[2]
  r_b         = sqrt(x_norm_b^2 + y_norm_b^2)

  inicenter_c = SVector(zero(delta), 0.65)
  x_norm_c    = x[1] - inicenter_c[1]
  y_norm_c    = x[2] - inicenter_c[2]
  r_c         = sqrt(x_norm_c^2 + y_norm_c^2)

  inicenter_d = SVector(zero(delta), -0.65)
  x_norm_d    = x[1] - inicenter_d[1]
  y_norm_d    = x[2] - inicenter_d[2]
  r_d         = sqrt(x_norm_d^2 + y_norm_d^2)

  inicenter_e = SVector(zero(delta), -1.60)
  x_norm_e    = x[1] - inicenter_e[1]
  y_norm_e    = x[2] - inicenter_e[2]
  r_e         = sqrt(x_norm_e^2 + y_norm_e^2)

  inicenter_f = SVector(zero(delta), -2.55)
  x_norm_f    = x[1] - inicenter_f[1]
  y_norm_f    = x[2] - inicenter_f[2]
  r_f         = sqrt(x_norm_f^2 + y_norm_f^2)


  if (x[1] > 0.4)
    # Set up Region III
    rho1    = rho1_3
    rho2    = rho2_3
    rho3    = rho3_3
    rho4    = rho4_3
    rho5    = rho5_3
    rho6    = rho6_3
    rho7    = rho7_3
    v1      = v1_3
    v2      = v2_3
    p       = p_3
  elseif (r_a < 0.25)
    # Set up Region I
    rho1    = rho1_1a
    rho2    = rho2_1a
    rho3    = rho3_1a
    rho4    = rho4_1a
    rho5    = rho5_1a
    rho6    = rho6_1a
    rho7    = rho7_1a
    v1      = v1_1a
    v2      = v2_1a
    p       = p_1a
  elseif (r_b < 0.25)
    # Set up Region I
    rho1    = rho1_1b
    rho2    = rho2_1b
    rho3    = rho3_1b
    rho4    = rho4_1b
    rho5    = rho5_1b
    rho6    = rho6_1b
    rho7    = rho7_1b
    v1      = v1_1b
    v2      = v2_1b
    p       = p_1b
  elseif (r_c < 0.25)
    # Set up Region I
    rho1    = rho1_1c
    rho2    = rho2_1c
    rho3    = rho3_1c
    rho4    = rho4_1c
    rho5    = rho5_1c
    rho6    = rho6_1c
    rho7    = rho7_1c
    v1      = v1_1c
    v2      = v2_1c
    p       = p_1c
  elseif (r_d < 0.25)
    # Set up Region I
    rho1    = rho1_1d
    rho2    = rho2_1d
    rho3    = rho3_1d
    rho4    = rho4_1d
    rho5    = rho5_1d
    rho6    = rho6_1d
    rho7    = rho7_1d
    v1      = v1_1d
    v2      = v2_1d
    p       = p_1d
  elseif (r_e < 0.25)
    # Set up Region I
    rho1    = rho1_1e
    rho2    = rho2_1e
    rho3    = rho3_1e
    rho4    = rho4_1e
    rho5    = rho5_1e
    rho6    = rho6_1e
    rho7    = rho7_1e
    v1      = v1_1e
    v2      = v2_1e
    p       = p_1e
  elseif (r_f < 0.25)
    # Set up Region I
    rho1    = rho1_1f
    rho2    = rho2_1f
    rho3    = rho3_1f
    rho4    = rho4_1f
    rho5    = rho5_1f
    rho6    = rho6_1f
    rho7    = rho7_1f
    v1      = v1_1f
    v2      = v2_1f
    p       = p_1f
  else
    # Set up Region II
    rho1    = rho1_2
    rho2    = rho2_2
    rho3    = rho3_2
    rho4    = rho4_2
    rho5    = rho5_2
    rho6    = rho6_2
    rho7    = rho7_2
    v1      = v1_2
    v2      = v2_2
    p       = p_2
  end

  return prim2cons(SVector(v1, v2, p, rho1, rho2, rho3, rho4, rho5, rho6, rho7), equations)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations2D)

A for multicomponent adapted weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerMulticomponentEquations2D)
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
@inline function flux(u, orientation::Integer, equations::CompressibleEulerMulticomponentEquations2D)
  rho_v1, rho_v2, rho_e  = u

  rho = density(u, equations)

  v1    = rho_v1/rho
  v2    = rho_v2/rho
  gamma = totalgamma(u, equations)
  p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))

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


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerMulticomponentEquations2D)

Entropy conserving two-point flux by
- Ayoub Gouasmi, Karthik Duraisamy (2020)
  "Formulation of Entropy-Stable schemes for the multicomponent compressible Euler equations""
  arXiv:1904.00972v3 [math.NA] 4 Feb 2020
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMulticomponentEquations2D)
  # Unpack left and right state
  @unpack gammas, gas_constants, cv = equations
  rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr
  rhok_mean   = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i+3], u_rr[i+3]) for i in eachcomponent(equations))
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

  T_ll        = (rho_e_ll - 0.5 * rho_ll * (v1_ll^2 + v2_ll^2)) / help1_ll
  T_rr        = (rho_e_rr - 0.5 * rho_rr * (v1_rr^2 + v2_rr^2)) / help1_rr
  T           = 0.5 * (1.0/T_ll + 1.0/T_rr)
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


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerMulticomponentEquations2D)
  rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  # Calculate primitive variables and speed of sound
  rho_ll   = density(u_ll, equations)
  rho_rr   = density(u_rr, equations)
  gamma_ll = totalgamma(u_ll, equations)
  gamma_rr = totalgamma(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  p_ll = (gamma_ll - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(gamma_ll * p_ll / rho_ll)

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  p_rr = (gamma_rr - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(gamma_rr * p_rr / rho_rr)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end


@inline function max_abs_speeds(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho_v1, rho_v2, rho_e = u

  rho   = density(u, equations)
  v1    = rho_v1 / rho
  v2    = rho_v2 / rho

  gamma = totalgamma(u, equations)
  p     = (gamma - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2))
  c     = sqrt(gamma * p / rho)

  return (abs(v1) + c, abs(v2) + c, )
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho_v1, rho_v2, rho_e = u

  prim_rho = SVector{ncomponents(equations), real(equations)}(u[i+3] for i in eachcomponent(equations))

  rho   = density(u, equations)
  v1    = rho_v1 / rho
  v2    = rho_v2 / rho
  gamma = totalgamma(u, equations)
  p     = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))
  prim_other =  SVector{3, real(equations)}(v1, v2, p)

  return vcat(prim_other, prim_rho)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gammas, gas_constants = equations
  rho_v1, rho_v2, rho_e = u

  rho       = density(u, equations)

  v1        = rho_v1 / rho
  v2        = rho_v2 / rho
  v_square  = v1^2 + v2^2
  gamma     = totalgamma(u, equations)
  p         = (gamma - 1) * (rho_e - 0.5 * rho * v_square)
  s         = log(p) - gamma*log(rho)
  rho_p     = rho / p

  # Multicomponent stuff
  help1 = zero(v1)

  for i in eachcomponent(equations)
    help1 += u[i+3] * cv[i]
  end

  T         = (rho_e - 0.5 * rho * v_square) / (help1)

  entrop_rho  = SVector{ncomponents(equations), real(equations)}( -1.0 * (cv[i] * log(T) - gas_constants[i] * log(u[i+3])) + gas_constants[i] + cv[i] - (v_square / (2*T)) for i in eachcomponent(equations))

  w1        = v1/T
  w2        = v2/T
  w3        = -1.0/T

  entrop_other = SVector{3, real(equations)}(w1, w2, w3)

  return vcat(entrop_other, entrop_rho)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gammas = equations
  v1, v2, p = prim

  cons_rho = SVector{ncomponents(equations), real(equations)}(prim[i+3] for i in eachcomponent(equations))
  rho     = density(prim, equations)
  gamma   = totalgamma(prim, equations)

  rho_v1  = rho * v1
  rho_v2  = rho * v2
  rho_e   = p/(gamma-1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)

  cons_other = SVector{3, real(equations)}(rho_v1, rho_v2, rho_e)

  return vcat(cons_other, cons_rho)
end


"""
    totalgamma(u, equations::CompressibleEulerMulticomponentEquations2D)

Function that calculates the total gamma out of all partial gammas using the
partial density fractions as well as the partial specific heats at constant volume.
"""
@inline function totalgamma(u, equations::CompressibleEulerMulticomponentEquations2D)
  @unpack cv, gammas = equations

  help1 = zero(u[1])
  help2 = zero(u[1])

  for i in eachcomponent(equations)
    help1 += u[i+3] * cv[i] * gammas[i]
    help2 += u[i+3] * cv[i]
  end

  return help1/help2
end


@inline function density_pressure(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho_v1, rho_v2, rho_e = u

  rho          = density(u, equations)
  gamma        = totalgamma(u, equations)
  rho_times_p  = (gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))

  return rho_times_p
end


@inline function density(u, equations::CompressibleEulerMulticomponentEquations2D)
  rho = zero(u[1])

  for i in eachcomponent(equations)
    rho += u[i+3]
  end

  return rho
 end

 @inline function densities(u, v, equations::CompressibleEulerMulticomponentEquations2D)

  return SVector{ncomponents(equations), real(equations)}(u[i+3]*v for i in eachcomponent(equations))
 end


end # @muladd
