# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    AcousticPerturbationEquations2DAuxVars(v_mean_global, c_mean_global, rho_mean_global)

Alternative implementation of [`AcousticPerturbationEquations2D`](@ref) using auxiliary
variables.
"""
struct AcousticPerturbationEquations2DAuxVars{RealT <: Real} <:
       AbstractAcousticPerturbationEquations{2, 3}
    v_mean_global::SVector{2, RealT}
    c_mean_global::RealT
    rho_mean_global::RealT
end

function AcousticPerturbationEquations2DAuxVars(v_mean_global::NTuple{2, <:Real},
                                                c_mean_global::Real,
                                                rho_mean_global::Real)
    return AcousticPerturbationEquations2DAuxVars(SVector(v_mean_global), c_mean_global,
                                                  rho_mean_global)
end

function AcousticPerturbationEquations2DAuxVars(; v_mean_global::NTuple{2, <:Real},
                                                c_mean_global::Real,
                                                rho_mean_global::Real)
    return AcousticPerturbationEquations2DAuxVars(SVector(v_mean_global), c_mean_global,
                                                  rho_mean_global)
end

have_aux_node_vars(::AcousticPerturbationEquations2DAuxVars) = True()
n_aux_node_vars(::AcousticPerturbationEquations2DAuxVars) = 4

"""
    global_mean_vars(equations::AcousticPerturbationEquations2DAuxVars)

Returns the global mean variables stored in `equations`. This makes it easier
to define flexible initial conditions for problems with constant mean flow.
"""
function global_mean_vars(equations::AcousticPerturbationEquations2DAuxVars)
    return equations.v_mean_global[1], equations.v_mean_global[2],
           equations.c_mean_global,
           equations.rho_mean_global
end

"""
    initial_condition_constant(x, t, equations::AcousticPerturbationEquations2DAuxVars)

A constant initial condition where the state variables are zero and the mean flow is constant.
Uses the global mean values from `equations`.
"""
function initial_condition_constant(x, t,
                                    equations::AcousticPerturbationEquations2DAuxVars)
    v1_prime = 0
    v2_prime = 0
    p_prime_scaled = 0

    return SVector(v1_prime, v2_prime, p_prime_scaled)
end

"""
    initial_condition_convergence_test(x, t, equations::AcousticPerturbationEquations2DAuxVars)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref). Uses the global mean values from `equations`.
"""
function initial_condition_convergence_test(x, t,
                                            equations::AcousticPerturbationEquations2DAuxVars)
    RealT = eltype(x)
    a = 1
    c = 2
    L = 2
    f = 2.0f0 / L
    A = convert(RealT, 0.2)
    omega = 2 * convert(RealT, pi) * f
    init = c + A * sin(omega * (x[1] + x[2] - a * t))

    v1_prime = init
    v2_prime = init
    p_prime = init^2
    p_prime_scaled = p_prime / equations.c_mean_global^2

    return SVector(v1_prime, v2_prime, p_prime_scaled)
end

"""
    source_terms_convergence_test(u, x, t, equations::AcousticPerturbationEquations2DAuxVars)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref).
"""
function source_terms_convergence_test(u, aux, x, t,
                                       equations::AcousticPerturbationEquations2DAuxVars)
    v1_mean, v2_mean, c_mean, rho_mean = aux

    RealT = eltype(u)
    a = 1
    c = 2
    L = 2
    f = 2.0f0 / L
    A = convert(RealT, 0.2)
    omega = 2 * convert(RealT, pi) * f

    si, co = sincos(omega * (x[1] + x[2] - a * t))
    tmp = v1_mean + v2_mean - a

    du1 = du2 = A * omega * co * (2 * c / rho_mean + tmp + 2 / rho_mean * A * si)
    du3 = A * omega * co * (2 * c_mean^2 * rho_mean + 2 * c * tmp + 2 * A * tmp * si) /
          c_mean^2

    return SVector(du1, du2, du3)
end

"""
    initial_condition_gauss(x, t, equations::AcousticPerturbationEquations2DAuxVars)

A Gaussian pulse in a constant mean flow. Uses the global mean values from `equations`.
"""
function initial_condition_gauss(x, t,
                                 equations::AcousticPerturbationEquations2DAuxVars)
    v1_prime = 0
    v2_prime = 0
    p_prime = exp(-4 * (x[1]^2 + x[2]^2))
    p_prime_scaled = p_prime / equations.c_mean_global^2

    return SVector(v1_prime, v2_prime, p_prime_scaled)
end

"""
    boundary_condition_wall(u_inner, aux_inner, orientation, direction, x, t,
                            surface_flux_function,
                            equations::AcousticPerturbationEquations2DAuxVars)

Boundary conditions for a solid wall.
"""
function boundary_condition_wall(u_inner, aux_inner, orientation, direction, x, t,
                                 surface_flux_function,
                                 equations::AcousticPerturbationEquations2DAuxVars)
    # Boundary state is equal to the inner state except for the perturbed velocity. For boundaries
    # in the -x/+x direction, we multiply the perturbed velocity in the x direction by -1.
    # Similarly, for boundaries in the -y/+y direction, we multiply the perturbed velocity in the
    # y direction by -1
    if direction in (1, 2) # x direction
        u_boundary = SVector(-u_inner[1], u_inner[2], u_inner[3])
    else # y direction
        u_boundary = SVector(u_inner[1], -u_inner[2], u_inner[3])
    end

    # Calculate boundary flux
    if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
        flux = surface_flux_function(u_inner, u_boundary, aux_inner, aux_inner,
                                     orientation, equations)
    else # u_boundary is "left" of boundary, u_inner is "right" of boundary
        flux = surface_flux_function(u_boundary, u_inner, aux_inner, aux_inner,
                                     orientation, equations)
    end

    return flux
end

"""
    boundary_condition_slip_wall(u_inner, aux_inner, normal_direction, x, t,
                                 surface_flux_function,
                                 equations::AcousticPerturbationEquations2DAuxVars)

Use an orthogonal projection of the perturbed velocities to zero out the normal velocity
while retaining the possibility of a tangential velocity in the boundary state.
Further details are available in the paper:
- Marcus Bauer, Jürgen Dierke and Roland Ewert (2011)
  Application of a discontinuous Galerkin method to discretize acoustic perturbation equations
  [DOI: 10.2514/1.J050333](https://doi.org/10.2514/1.J050333)
"""
function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, x, t,
                                      surface_flux_function,
                                      equations::AcousticPerturbationEquations2DAuxVars)
    # normalize the outward pointing direction
    normal = normal_direction / norm(normal_direction)

    # compute the normal perturbed velocity
    u_normal = normal[1] * u_inner[1] + normal[2] * u_inner[2]

    # create the "external" boundary solution state
    u_boundary = SVector(u_inner[1] - 2 * u_normal * normal[1],
                         u_inner[2] - 2 * u_normal * normal[2],
                         u_inner[3])

    # calculate the boundary flux
    flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

    return flux
end

# Calculate 1D flux for a single point
@inline function flux(u, aux, orientation::Integer,
                      equations::AcousticPerturbationEquations2DAuxVars)
    v1_prime, v2_prime, p_prime_scaled = u
    v1_mean, v2_mean, c_mean, rho_mean = aux

    # Calculate flux for conservative state variables
    RealT = eltype(u)
    if orientation == 1
        f1 = v1_mean * v1_prime + v2_mean * v2_prime +
             c_mean^2 * p_prime_scaled / rho_mean
        f2 = zero(RealT)
        f3 = rho_mean * v1_prime + v1_mean * p_prime_scaled
    else
        f1 = zero(RealT)
        f2 = v1_mean * v1_prime + v2_mean * v2_prime +
             c_mean^2 * p_prime_scaled / rho_mean
        f3 = rho_mean * v2_prime + v2_mean * p_prime_scaled
    end

    return SVector(f1, f2, f3)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                                     equations::AcousticPerturbationEquations2DAuxVars)
    # Calculate v = v_prime + v_mean
    v_prime_ll = u_ll[orientation]
    v_prime_rr = u_rr[orientation]
    v_mean_ll = aux_ll[orientation]
    v_mean_rr = aux_rr[orientation]

    v_ll = v_prime_ll + v_mean_ll
    v_rr = v_prime_rr + v_mean_rr

    c_mean_ll = aux_ll[3]
    c_mean_rr = aux_rr[3]

    return max(abs(v_ll), abs(v_rr)) + max(c_mean_ll, c_mean_rr)
end

# Less "cautious", i.e., less overestimating `λ_max` compared to `max_abs_speed_naive`
@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr, orientation::Integer,
                               equations::AcousticPerturbationEquations2DAuxVars)
    # Calculate v = v_prime + v_mean
    v_prime_ll = u_ll[orientation]
    v_prime_rr = u_rr[orientation]
    v_mean_ll = aux_ll[orientation]
    v_mean_rr = aux_rr[orientation]

    v_ll = v_prime_ll + v_mean_ll
    v_rr = v_prime_rr + v_mean_rr

    c_mean_ll = aux_ll[3]
    c_mean_rr = aux_rr[3]

    return max(abs(v_ll) + c_mean_ll, abs(v_rr) + c_mean_rr)
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, aux, normal_direction::AbstractVector,
                      equations::AcousticPerturbationEquations2DAuxVars)
    v1_prime, v2_prime, p_prime_scaled = u
    v1_mean, v2_mean, c_mean, rho_mean = aux

    f1 = normal_direction[1] * (v1_mean * v1_prime + v2_mean * v2_prime +
          c_mean^2 * p_prime_scaled / rho_mean)
    f2 = normal_direction[2] * (v1_mean * v1_prime + v2_mean * v2_prime +
          c_mean^2 * p_prime_scaled / rho_mean)
    f3 = (normal_direction[1] * (rho_mean * v1_prime + v1_mean * p_prime_scaled)
          +
          normal_direction[2] * (rho_mean * v2_prime + v2_mean * p_prime_scaled))

    return SVector(f1, f2, f3)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, aux_ll, aux_rr,
                                     normal_direction::AbstractVector,
                                     equations::AcousticPerturbationEquations2DAuxVars)
    # Calculate v = v_prime + v_mean
    v_prime_ll = normal_direction[1] * u_ll[1] + normal_direction[2] * u_ll[2]
    v_prime_rr = normal_direction[1] * u_rr[1] + normal_direction[2] * u_rr[2]
    v_mean_ll = normal_direction[1] * aux_ll[1] + normal_direction[2] * aux_ll[2]
    v_mean_rr = normal_direction[1] * aux_rr[1] + normal_direction[2] * aux_rr[2]

    v_ll = v_prime_ll + v_mean_ll
    v_rr = v_prime_rr + v_mean_rr

    c_mean_ll = aux_ll[3]
    c_mean_rr = aux_rr[3]

    # The v_normals are already scaled by the norm
    return (max(abs(v_ll), abs(v_rr)) +
            max(c_mean_ll, c_mean_rr) * norm(normal_direction))
end

# Less "cautious", i.e., less overestimating `λ_max` compared to `max_abs_speed_naive`
@inline function max_abs_speed(u_ll, u_rr, aux_ll, aux_rr,
                               normal_direction::AbstractVector,
                               equations::AcousticPerturbationEquations2DAuxVars)
    # Calculate v = v_prime + v_mean
    v_prime_ll = normal_direction[1] * u_ll[1] + normal_direction[2] * u_ll[2]
    v_prime_rr = normal_direction[1] * u_rr[1] + normal_direction[2] * u_rr[2]
    v_mean_ll = normal_direction[1] * aux_ll[1] + normal_direction[2] * aux_ll[2]
    v_mean_rr = normal_direction[1] * aux_rr[1] + normal_direction[2] * aux_rr[2]

    v_ll = v_prime_ll + v_mean_ll
    v_rr = v_prime_rr + v_mean_rr

    c_mean_ll = aux_ll[3]
    c_mean_rr = aux_rr[3]

    norm_ = norm(normal_direction)
    # The v_normals are already scaled by the norm
    return max(abs(v_ll) + c_mean_ll * norm_, abs(v_rr) + c_mean_rr * norm_)
end

@inline have_constant_speed(::AcousticPerturbationEquations2DAuxVars) = False()

@inline function max_abs_speeds(u, aux,
                                equations::AcousticPerturbationEquations2DAuxVars)
    v1_mean, v2_mean, c_mean, _ = aux

    return abs(v1_mean) + c_mean, abs(v2_mean) + c_mean
end

function varnames(::typeof(cons2cons), ::AcousticPerturbationEquations2DAuxVars)
    ("v1_prime", "v2_prime", "p_prime_scaled")
end

# Convenience functions for retrieving state variables and mean variables
function cons2state(u, aux, ::AcousticPerturbationEquations2DAuxVars)
    return SVector(u[1], u[2], u[3])
end

function varnames(::typeof(cons2state), ::AcousticPerturbationEquations2DAuxVars)
    ("v1_prime", "v2_prime", "p_prime_scaled")
end

function cons2aux(u, aux, equations::AcousticPerturbationEquations2D)
    return SVector(aux[1], aux[2], aux[3], aux[4])
end

function varnames(::typeof(cons2aux), ::AcousticPerturbationEquations2DAuxVars)
    ("v1_mean", "v2_mean", "c_mean", "rho_mean")
end

# Convert conservative variables to primitive
@inline function cons2prim(u, aux, equations::AcousticPerturbationEquations2DAuxVars)
    p_prime_scaled = u[3]
    c_mean = aux[3]
    p_prime = p_prime_scaled * c_mean^2

    return SVector(u[1], u[2], p_prime)
end

function varnames(::typeof(cons2prim), ::AbstractAcousticPerturbationEquations{2})
    ("v1_prime", "v2_prime", "p_prime",
     "v1_mean", "v2_mean", "c_mean", "rho_mean")
end

# Convert primitive variables to conservative
@inline function prim2cons(u, aux, equations::AcousticPerturbationEquations2DAuxVars)
    p_prime = u[3]
    c_mean = aux[3]
    p_prime_scaled = p_prime / c_mean^2

    return SVector(u[1], u[2], p_prime_scaled)
end

# Convert conservative variables to entropy variables
@inline cons2entropy(u, aux, equations::AcousticPerturbationEquations2DAuxVars) = u
end # @muladd
