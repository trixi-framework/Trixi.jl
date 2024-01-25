# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function varnames(::typeof(cons2cons),
                  equations::AbstractCompressibleEulerMulticomponentEquations{2})
    cons = ("rho_v1", "rho_v2", "rho_e")
    rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (cons..., rhos...)
end

function varnames(::typeof(cons2prim),
                  equations::AbstractCompressibleEulerMulticomponentEquations{2})
    prim = ("v1", "v2", "p")
    rhos = ntuple(n -> "rho" * string(n), Val(ncomponents(equations)))
    return (prim..., rhos...)
end

# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer,
                      equations::AbstractCompressibleEulerMulticomponentEquations{2})
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = pressure(u, equations)

    if orientation == 1
        f_rho = u[4:end] .* v1
        f1 = rho_v1 * v1 + p
        f2 = rho_v2 * v1
        f3 = (rho_e + p) * v1
    else
        f_rho = u[4:end] .* v2
        f1 = rho_v1 * v2
        f2 = rho_v2 * v2 + p
        f3 = (rho_e + p) * v2
    end

    f_other = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
end

# Calculate 1D flux for a single point
@inline function flux(u, normal_direction::AbstractVector,
                      equations::AbstractCompressibleEulerMulticomponentEquations{2})
    rho_v1, rho_v2, rho_e = u

    rho = density(u, equations)

    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
    p = pressure(u, equations)

    f_rho = u[4:end] .* v_normal
    f1 = rho_v1 * v_normal + p * normal_direction[1]
    f2 = rho_v2 * v_normal + p * normal_direction[2]
    f3 = (rho_e + p) * v_normal

    f_other = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
end

@inline function rotate_to_x(u, normal_vector,
                             equations::AbstractCompressibleEulerMulticomponentEquations{2})
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D rotation matrix with normal and tangent directions of the form
    # [ n_1  n_2  0    0;
    #   t_1  t_2  0    0;
    #   0    0    1    0
    #   0    0    0    1]
    # where t_1 = -n_2 and t_2 = n_1

    densities = @view u[4:end]
    return SVector(c * u[1] + s * u[2],
                   -s * u[1] + c * u[2],
                   u[3],
                   densities...)
end

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector,
                               equations::AbstractCompressibleEulerMulticomponentEquations{2})
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D back-rotation matrix with normal and tangent directions of the form
    # [ n_1  t_1  0   0;
    #   n_2  t_2  0   0;
    #   0    0    1   0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    densities = @view u[4:end]
    return SVector(c * u[1] - s * u[2],
                   s * u[1] + c * u[2],
                   u[3],
                   densities...)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::AbstractCompressibleEulerMulticomponentEquations{2})

Determine the boundary numerical surface flux for a slip wall condition.
Imposes a zero normal velocity at the wall.
Density is taken from the internal solution state and pressure is computed as an
exact solution of a 1D Riemann problem. Further details about this boundary state
are available in the paper:
- J. J. W. van der Vegt and H. van der Ven (2002)
  Slip flow boundary conditions in discontinuous Galerkin discretizations of
  the Euler equations of gas dynamics
  [PDF](https://reports.nlr.nl/bitstream/handle/10921/692/TP-2002-300.pdf?sequence=1)

Details about the 1D pressure Riemann solution can be found in Section 6.3.3 of the book
- Eleuterio F. Toro (2009)
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
  3rd edition
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

Should be used together with [`UnstructuredMesh2D`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t, surface_flux_function,
                                              equations::AbstractCompressibleEulerMulticomponentEquations{2})
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal = normal_direction / norm_

    # rotate the internal solution state
    u_local = rotate_to_x(u_inner, normal, equations)

    # compute the primitive variables
    v_normal, v_tangent, p_local = cons2prim(u_local, equations)

    rho_local = density(u_local, equations)
    gamma = totalgamma(u_inner, equations)

    # Get the solution of the pressure Riemann problem
    if v_normal <= 0.0
        sound_speed = sqrt(gamma * p_local / rho_local) # local sound speed
        p_star = p_local *
                 (1 + 0.5 * (gamma - 1) * v_normal / sound_speed)^(2 * gamma *
                                                                   inv(gamma - 1))
    else # v_normal > 0.0
        A = 2 / ((gamma + 1) * rho_local)
        B = p_local * (gamma - 1) / (gamma + 1)
        p_star = p_local +
                 0.5 * v_normal / A *
                 (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
    end

    return SVector(p_star * normal[1],
                   p_star * normal[2],
                   zeros(eltype(u_inner), ncomponents(equations) + 1)...) * norm_
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                surface_flux_function, equations::AbstractCompressibleEulerMulticomponentEquations{2})

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::AbstractCompressibleEulerMulticomponentEquations{2})
    # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
    # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
    if isodd(direction)
        boundary_flux = -boundary_condition_slip_wall(u_inner, -normal_direction,
                                                      x, t, surface_flux_function,
                                                      equations)
    else
        boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
                                                     x, t, surface_flux_function,
                                                     equations)
    end

    return boundary_flux
end

# Convert conservative variables to primitive
@inline function cons2prim(u,
                           equations::AbstractCompressibleEulerMulticomponentEquations{2})
    rho_v1, rho_v2, rho_e = u

    prim_rho = SVector{ncomponents(equations), real(equations)}(u[i + 3]
                                                                for i in eachcomponent(equations))

    rho = density(u, equations)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    p = pressure(u, equations)
    prim_other = SVector{3, real(equations)}(v1, v2, p)

    return vcat(prim_other, prim_rho)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim,
                           equations::AbstractCompressibleEulerMulticomponentEquations{2})
    v1, v2, p = prim

    cons_rho = SVector{ncomponents(equations), real(equations)}(prim[i + 3]
                                                                for i in eachcomponent(equations))
    rho = density(prim, equations)
    gamma = totalgamma(prim, equations)

    rho_v1 = rho * v1
    rho_v2 = rho * v2

    rho_e = p / (gamma - 1) + 0.5 * (rho_v1 * v1 + rho_v2 * v2)

    cons_other = SVector{3, real(equations)}(rho_v1, rho_v2, rho_e)

    return vcat(cons_other, cons_rho)
end

"""
    density(u, equations::AbstractCompressibleEulerMulticomponentEquations{2})

Function that calculates the overall density.
"""
@inline function density(u,
                         equations::AbstractCompressibleEulerMulticomponentEquations{2})
    rho = zero(u[1])
    for i in eachcomponent(equations)
        rho += u[i + 3]
    end
    return rho
end

"""
    totalgamma(u, equations::AbstractCompressibleEulerMulticomponentEquations{2})

Function that calculates the total gamma out of all partial gammas.
This function has to be implemented for subtypes!
"""
function totalgamma end

"""
    temperature(u, equations::CompressibleEulerMulticomponentEquations2D)

Calculate temperature.
This function has to be implemented for subtypes!
"""
function temperature end

"""
    pressure(u, equations::AbstractCompressibleEulerMulticomponentEquations{2})

Function that calculates the overall pressure.
"""
@inline function pressure(u,
                          equations::AbstractCompressibleEulerMulticomponentEquations{2})
    rho_v1, rho_v2, rho_e = u
    rho = density(u, equations)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    gamma = totalgamma(u, equations)

    p = (gamma - 1) * (rho_e - 0.5 * rho * (v1^2 + v2^2))
    return p
end

"""
    density_gas_constant(u, equations::AbstractCompressibleEulerMulticomponentEquations{2})

Function that calculates overall density times overall gas constant.
This has to be implemented for subtypes!
"""
function density_gas_constant end

"""
    initial_condition_convergence_test(x, t, equations::AbstractCompressibleEulerMulticomponentEquations{2})

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t,
                                            equations::AbstractCompressibleEulerMulticomponentEquations{2})
    c = 2
    A = 0.1
    L = 2
    f = 1 / L
    omega = 2 * pi * f
    ini = c + A * sin(omega * (x[1] + x[2] - t))

    v1 = 1.0
    v2 = 1.0

    rho = ini

    # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1)
    prim_rho = SVector{ncomponents(equations), real(equations)}(2^(i - 1) * (1 - 2) /
                                                                (1 -
                                                                 2^ncomponents(equations)) *
                                                                rho
                                                                for i in eachcomponent(equations))

    prim1 = rho * v1
    prim2 = rho * v2
    prim3 = rho^2

    prim_other = SVector{3, real(equations)}(prim1, prim2, prim3)

    return vcat(prim_other, prim_rho)
end

"""
    source_terms_convergence_test(u, x, t, equations::AbstractCompressibleEulerMulticomponentEquations{2})

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t,
                                               equations::AbstractCompressibleEulerMulticomponentEquations{2})
    # Same settings as in `initial_condition`
    c = 2
    A = 0.1
    L = 2
    f = 1 / L
    omega = 2 * pi * f

    gamma = totalgamma(u, equations)

    x1, x2 = x
    si, co = sincos((x1 + x2 - t) * omega)
    tmp1 = co * A * omega
    tmp2 = si * A
    tmp3 = gamma - 1
    tmp4 = (2 * c - 1) * tmp3
    tmp5 = (2 * tmp2 * gamma - 2 * tmp2 + tmp4 + 1) * tmp1
    tmp6 = tmp2 + c

    # Here we compute an arbitrary number of different rhos. (one rho is double the next rho while the sum of all rhos is 1
    du_rho = SVector{ncomponents(equations), real(equations)}(2^(i - 1) * (1 - 2) /
                                                              (1 -
                                                               2^ncomponents(equations)) *
                                                              tmp1
                                                              for i in eachcomponent(equations))

    du1 = tmp5
    du2 = tmp5
    du3 = 2 * ((tmp6 - 1.0) * tmp3 + tmp6 * gamma) * tmp1

    du_other = SVector{3, real(equations)}(du1, du2, du3)

    return vcat(du_other, du_rho)
end

"""
    initial_condition_weak_blast_wave(x, t, equations::AbstractCompressibleEulerMulticomponentEquations{2})

A for multicomponent adapted weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t,
                                           equations::AbstractCompressibleEulerMulticomponentEquations{2})
    # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    prim_rho = SVector{ncomponents(equations), real(equations)}(r > 0.5 ?
                                                                2^(i - 1) * (1 - 2) /
                                                                (1 -
                                                                 2^ncomponents(equations)) *
                                                                1.0 :
                                                                2^(i - 1) * (1 - 2) /
                                                                (1 -
                                                                 2^ncomponents(equations)) *
                                                                1.1691
                                                                for i in eachcomponent(equations))

    v1 = r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2 = r > 0.5 ? 0.0 : 0.1882 * sin_phi
    p = r > 0.5 ? 1.0 : 1.245

    prim_other = SVector{3, real(equations)}(v1, v2, p)

    return prim2cons(vcat(prim_other, prim_rho), equations)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::AbstractCompressibleEulerMulticomponentEquations{2})
    rho_v1_ll, rho_v2_ll = u_ll
    rho_v1_rr, rho_v2_rr = u_rr

    # Get the density and gas gamma
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)
    gamma_ll = totalgamma(u_ll, equations)
    gamma_rr = totalgamma(u_rr, equations)

    # Get the velocities based on direction
    if orientation == 1
        v_ll = rho_v1_ll / rho_ll
        v_rr = rho_v1_rr / rho_rr
    else # orientation == 2
        v_ll = rho_v2_ll / rho_ll
        v_rr = rho_v2_rr / rho_rr
    end

    # Compute the sound speeds on the left and right
    p_ll = pressure(u_ll, equations)
    c_ll = sqrt(gamma_ll * p_ll / rho_ll)
    p_rr = pressure(u_rr, equations)
    c_rr = sqrt(gamma_rr * p_rr / rho_rr)

    λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end

@inline function max_abs_speeds(u,
                                equations::AbstractCompressibleEulerMulticomponentEquations{2})
    rho_v1, rho_v2 = u

    rho = density(u, equations)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho

    gamma = totalgamma(u, equations)
    p = pressure(u, equations)
    c = sqrt(gamma * p / rho)

    return (abs(v1) + c, abs(v2) + c)
end

"""
flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
             equations::AbstractCompressibleEulerMulticomponentEquations{2})

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
@inline function flux_ranocha(u_ll, u_rr, orientation::Integer,
                              equations::AbstractCompressibleEulerMulticomponentEquations{2})
    # Unpack left and right state
    rho_v1_ll, rho_v2_ll = u_ll
    rho_v1_rr, rho_v2_rr = u_rr
    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 3],
                                                                         u_rr[i + 3])
                                                                 for i in eachcomponent(equations))

    # Iterating over all partial densities
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    # Calculating gamma
    gamma = totalgamma(0.5 * (u_ll + u_rr), equations)
    inv_gamma_minus_one = 1 / (gamma - 1)

    # extract velocities
    v1_ll = rho_v1_ll / rho_ll
    v1_rr = rho_v1_rr / rho_rr
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_ll = rho_v2_ll / rho_ll
    v2_rr = rho_v2_rr / rho_rr
    v2_avg = 0.5 * (v2_ll + v2_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr)

    # helpful variables
    enth_ll = density_gas_constant(u_ll)
    enth_rr = density_gas_constant(u_rr)

    # temperature and pressure
    T_ll = temperature(u_ll, equations)
    T_rr = temperature(u_rr, equations)
    p_ll = T_ll * enth_ll
    p_rr = T_rr * enth_rr
    p_avg = 0.5 * (p_ll + p_rr)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)

    f_rho_sum = zero(T_rr)
    if orientation == 1
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v1_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            f_rho_sum += f_rho[i]
        end
        f1 = f_rho_sum * v1_avg + p_avg
        f2 = f_rho_sum * v2_avg
        f3 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
             0.5 * (p_ll * v1_rr + p_rr * v1_ll)
    else
        f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * v2_avg
                                                                 for i in eachcomponent(equations))
        for i in eachcomponent(equations)
            f_rho_sum += f_rho[i]
        end
        f1 = f_rho_sum * v1_avg
        f2 = f_rho_sum * v2_avg + p_avg
        f3 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
             0.5 * (p_ll * v2_rr + p_rr * v2_ll)
    end

    # momentum and energy flux
    f_other = SVector{3, real(equations)}(f1, f2, f3)

    return vcat(f_other, f_rho)
end

@inline function flux_ranocha(u_ll, u_rr, normal_direction::AbstractVector,
                              equations::AbstractCompressibleEulerMulticomponentEquations{2})
    # Unpack left and right state
    rho_v1_ll, rho_v2_ll = u_ll
    rho_v1_rr, rho_v2_rr = u_rr
    rhok_mean = SVector{ncomponents(equations), real(equations)}(ln_mean(u_ll[i + 3],
                                                                         u_rr[i + 3])
                                                                 for i in eachcomponent(equations))

    # Iterating over all partial densities
    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    # Calculating gamma
    gamma = totalgamma(0.5 * (u_ll + u_rr), equations)
    inv_gamma_minus_one = 1 / (gamma - 1)

    # extract velocities
    v1_ll = rho_v1_ll / rho_ll
    v1_rr = rho_v1_rr / rho_rr
    v1_avg = 0.5 * (v1_ll + v1_rr)
    v2_ll = rho_v2_ll / rho_ll
    v2_rr = rho_v2_rr / rho_rr
    v2_avg = 0.5 * (v2_ll + v2_rr)
    velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr)
    v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # helpful variables
    enth_ll = density_gas_constant(u_ll)
    enth_rr = density_gas_constant(u_rr)

    # temperature and pressure
    T_ll = temperature(u_ll, equations)
    T_rr = temperature(u_rr, equations)
    p_ll = T_ll * enth_ll
    p_rr = T_rr * enth_rr
    p_avg = 0.5 * (p_ll + p_rr)
    inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)

    f_rho_sum = zero(T_rr)
    f_rho = SVector{ncomponents(equations), real(equations)}(rhok_mean[i] * 0.5 *
                                                             (v_dot_n_ll + v_dot_n_rr)
                                                             for i in eachcomponent(equations))
    for i in eachcomponent(equations)
        f_rho_sum += f_rho[i]
    end
    f1 = f_rho_sum * v1_avg + p_avg * normal_direction[1]
    f2 = f_rho_sum * v2_avg + p_avg * normal_direction[2]
    f3 = f_rho_sum * (velocity_square_avg + inv_rho_p_mean * inv_gamma_minus_one) +
         0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll)

    # momentum and energy flux
    f_other = SVector(f1, f2, f3)

    return vcat(f_other, f_rho)
end

@inline function (flux_lmars::FluxLMARS)(u_ll, u_rr, normal_direction::AbstractVector,
                                         equations::AbstractCompressibleEulerMulticomponentEquations{2})
    c = flux_lmars.speed_of_sound

    # Unpack left and right state
    v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
    v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

    rho_ll = density(u_ll, equations)
    rho_rr = density(u_rr, equations)

    v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
    v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

    # Note that this is the same as computing v_ll and v_rr with a normalized normal vector
    # and then multiplying v by `norm_` again, but this version is slightly faster.
    norm_ = norm(normal_direction)

    rho = 0.5 * (rho_ll + rho_rr)
    p = 0.5 * (p_ll + p_rr) - 0.5 * c * rho * (v_rr - v_ll) / norm_
    v = 0.5 * (v_ll + v_rr) - 1 / (2 * c * rho) * (p_rr - p_ll) * norm_

    # We treat the energy term analogous to the potential temperature term in the paper
    # by Chen et al., i.e. we use p_ll and p_rr, and not p
    if v >= 0
        f1, f2, f3 = u_ll * v
        f3 = f3 + p_ll * v
        # density fluxes
        f_rho = SVector{ncomponents(equations), real(equations)}(u_ll[ncomponents(equations) + i] *
                                                                 v
                                                                 for i in eachcomponent(equations))
    else
        f1, f2, f3 = u_rr * v
        f3 = f3 + p_rr * v
        # density fluxes
        f_rho = SVector{ncomponents(equations), real(equations)}(u_rr[ncomponents(equations) + i] *
                                                                 v
                                                                 for i in eachcomponent(equations))
    end
    f1 = f1 + p * normal_direction[1]
    f2 = f2 + p * normal_direction[2]

    # momentum and energy flux
    f_other = SVector(f1, f2, f3)

    return vcat(f_other, f_rho)
end
end # @muladd
