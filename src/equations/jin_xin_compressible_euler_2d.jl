# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    JinXinEquations

TODO: Write a proper docstring
"""
struct JinXinEquations{NDIMS, NVARS, NVARS_BASE,
                       EquationsBase <: AbstractEquations{NDIMS, NVARS_BASE},
                       RealT <: Real} <:
       AbstractJinXinEquations{NDIMS, NVARS}
    equations_base::EquationsBase
    # relaxation parameter of the Jin Xin relaxation model
    # The relaxed equations should converge to the original equations
    # as the relaxation parameter epsilon -> 0
    eps_relaxation     :: RealT
    eps_relaxation_inv :: RealT

    # velocity parameters of the Jin Xin relaxation model
    # They need to satisfy the subcharacteristic condition, i.e., they
    # must be upper bounds of the characteristic speeds of the original
    # equations
    velocities::NTuple{NDIMS, SVector{NVARS_BASE, RealT}}
    sqrt_velocities::NTuple{NDIMS, SVector{NVARS_BASE, RealT}}
    sqrt_velocities_inv::NTuple{NDIMS, SVector{NVARS_BASE, RealT}}
end

function JinXinEquations(equations_base, eps_relaxation, velocities)
    sqrt_velocities = map(velocities) do v
        sqrt_v = sqrt.(v)
        return sqrt_v
    end
    sqrt_velocities_inv = map(sqrt_velocities) do sv
        sqrt_vi = inv.(sv)
        return sqrt_vi
    end

    NDIMS = ndims(equations_base)
    NVARS_BASE = nvariables(equations_base)
    RealT = promote_type(typeof(eps_relaxation), eltype(eltype(velocities)))
    JinXinEquations{NDIMS, (NDIMS + 1) * NVARS_BASE, NVARS_BASE, typeof(equations_base),
                    RealT}(equations_base, eps_relaxation, inv(eps_relaxation),
                           velocities, sqrt_velocities, sqrt_velocities_inv)
end

function varnames(func::typeof(cons2cons), equations::JinXinEquations)
    basenames = varnames(func, equations.equations_base)

    if ndims(equations) == 1
        velocities_1 = ntuple(n -> "v1_" * string(n),
                              Val(nvariables(equations.equations_base)))
        return (basenames..., velocities_1...)
    elseif ndims(equations) == 2
        velocities_1 = ntuple(n -> "v1_" * string(n),
                              Val(nvariables(equations.equations_base)))
        velocities_2 = ntuple(n -> "v2_" * string(n),
                              Val(nvariables(equations.equations_base)))
        return (basenames..., velocities_1..., velocities_2...)
    elseif ndims(equations) == 3
        velocities_1 = ntuple(n -> "v1_" * string(n),
                              Val(nvariables(equations.equations_base)))
        velocities_2 = ntuple(n -> "v2_" * string(n),
                              Val(nvariables(equations.equations_base)))
        velocities_3 = ntuple(n -> "v3_" * string(n),
                              Val(nvariables(equations.equations_base)))
        return (basenames..., velocities_1..., velocities_2..., velocities_3...)
    else
        throw(ArgumentError("Number of dimensions $(ndims(equations)) not supported"))
    end
end

function varnames(::typeof(cons2prim), equations::JinXinEquations)
    # TODO: Jin Xin
    varnames(cons2cons, equations)
end

# Set initial conditions at physical location `x` for time `t`
struct InitialConditionJinXin{IC}
    initial_condition::IC
end

@inline function (ic::InitialConditionJinXin)(x, t, equations::JinXinEquations)
    eq_base = equations.equations_base
    u = ic.initial_condition(x, t, eq_base)

    if ndims(equations) == 1
        v1 = flux(u, 1, eq_base)
        return SVector(u..., v1...)
    elseif ndims(equations) == 2
        v1 = flux(u, 1, eq_base)
        v2 = flux(u, 2, eq_base)
        return SVector(u..., v1..., v2...)
    else
        v1 = flux(u, 1, eq_base)
        v2 = flux(u, 2, eq_base)
        v3 = flux(u, 3, eq_base)
        return SVector(u..., v1..., v2..., v3...)
    end
end

# Pre-defined source terms should be implemented as
# @inline function source_terms_JinXin_Relaxation(u, x, t,
#                                                equations::JinXinEquations)

#     # relaxation parameter
#     eps_inv = equations.eps_relaxation_inv

#     # compute compressible Euler fluxes
#     u_cons = SVector(u[1], u[2], u[3], u[4])
#     eq_relax = equations.equations_base
#     vu = flux(u_cons,1,eq_relax)
#     wu = flux(u_cons,2,eq_relax)

#     # compute relaxation terms
#     du1 = 0.0
#     du2 = 0.0
#     du3 = 0.0
#     du4 = 0.0
#     du5 = -eps_inv * (u[5] - vu[1])
#     du6 = -eps_inv * (u[6] - vu[2])
#     du7 = -eps_inv * (u[7] - vu[3])
#     du8 = -eps_inv * (u[8] - vu[4])
#     du9 = -eps_inv * (u[9] - wu[1])
#     du10= -eps_inv * (u[10]- wu[2])
#     du11= -eps_inv * (u[11]- wu[3])
#     du12= -eps_inv * (u[12]- wu[4])

#     return SVector(du1, du2, du3, du4, du5, du6, du7, du8, du9, du10, du11, du12)
# end

function get_block_components(u, n, equations::JinXinEquations)
    nvars_base = nvariables(equations.equations_base)
    return SVector(ntuple(i -> u[i + (n - 1) * nvars_base], Val(nvars_base)))
end

# Calculate 1D flux in for a single point
# TODO: Implement 1D and 3D
@inline function flux(u, orientation::Integer, equations::JinXinEquations{2})
    if orientation == 1
        u_base = get_block_components(u, 1, equations)
        fluxes = get_block_components(u, 2, equations)
        velocities = equations.velocities[1]
        return SVector(fluxes..., (velocities .* u_base)..., zero(u_base)...)
    else # orientation == 2
        u_base = get_block_components(u, 1, equations)
        fluxes = get_block_components(u, 3, equations)
        velocities = equations.velocities[2]
        return SVector(fluxes..., zero(u_base)..., (velocities .* u_base)...)
    end
end

@inline function flux(u, orientation::Integer, equations::JinXinEquations{1})
    u_base = get_block_components(u, 1, equations)
    fluxes = get_block_components(u, 2, equations)
    velocities = equations.velocities[1]
    return SVector(fluxes..., (velocities .* u_base)...)
end

# TODO: Implement 1D and 3D
@inline function flux_upwind(u_ll, u_rr, orientation::Integer,
                             equations::JinXinEquations{2})
    u_ll_base = get_block_components(u_ll, 1, equations)
    u_rr_base = get_block_components(u_rr, 1, equations)

    if orientation == 1
        sqrt_velocities = equations.sqrt_velocities[1]
        f_ll_base = get_block_components(u_ll, 2, equations)
        f_rr_base = get_block_components(u_rr, 2, equations)
        dissipation = SVector((sqrt_velocities .* (u_rr_base - u_ll_base))...,
                              (sqrt_velocities .* (f_rr_base - f_ll_base))...,
                              zero(u_ll_base)...)
    else # orientation == 2
        sqrt_velocities = equations.sqrt_velocities[2]
        f_ll_base = get_block_components(u_ll, 3, equations)
        f_rr_base = get_block_components(u_rr, 3, equations)
        dissipation = SVector((sqrt_velocities .* (u_rr_base - u_ll_base))...,
                              zero(u_ll_base)...,
                              (sqrt_velocities .* (f_rr_base - f_ll_base))...)
    end

    return 0.5f0 * (flux(u_ll, orientation, equations) +
            flux(u_rr, orientation, equations) - dissipation)
end

@inline function flux_upwind(u_ll, u_rr, orientation::Integer,
                             equations::JinXinEquations{1})
    u_ll_base = get_block_components(u_ll, 1, equations)
    u_rr_base = get_block_components(u_rr, 1, equations)

    sqrt_velocities = equations.sqrt_velocities[1]
    f_ll_base = get_block_components(u_ll, 2, equations)
    f_rr_base = get_block_components(u_rr, 2, equations)
    dissipation = SVector((sqrt_velocities .* (u_rr_base - u_ll_base))...,
                          #   (sqrt_velocities .* (f_rr_base + f_ll_base))..., @ranocha: is this correct?
                          (sqrt_velocities .* (f_rr_base - f_ll_base))...)
    return 0.5f0 * (flux(u_ll, orientation, equations) +
            flux(u_rr, orientation, equations) - dissipation)
end

@inline function max_abs_speeds(u, equations::JinXinEquations{2})
    return ntuple(Val(ndims(equations))) do n
        maximum(equations.sqrt_velocities[n])
    end
end

@inline function max_abs_speeds(u, equations::JinXinEquations{1})
    return ntuple(Val(ndims(equations))) do n
        # maximum(equations.sqrt_velocities_inv[n]) @ranocha: is this correct?
        maximum(equations.sqrt_velocities[n])
    end
end

# TODO: not correct yet!!
# Convert conservative variables to primitive
@inline cons2prim(u, equations::JinXinEquations) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::JinXinEquations) = u
@inline entropy2cons(u, equations::JinXinEquations) = u

@inline prim2cons(u, equations::JinXinEquations) = u
end # @muladd
