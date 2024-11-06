# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

const TRIXI_UUID = UUID("a7f1ee26-1774-49b1-8366-f1abc58fbfcb")

"""
    Trixi.set_polyester!(toggle::Bool; force = true)

Toggle the usage of [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl) for multithreading.
By default, Polyester.jl is enabled, but it can
be useful for performance comparisons to switch to the Julia core backend.

This does not fully disable Polyester.jl,
buy only its use as part of Trixi.jl's `@threaded` macro.
"""
function set_polyester!(toggle::Bool; force = true)
    set_preferences!(TRIXI_UUID, "polyester" => toggle, force = force)
    @info "Please restart Julia and reload Trixi.jl for the `polyester` change to take effect"
end

"""
    Trixi.set_sqrt_type!(type; force = true)

Set the `type` of the square root function to be used in Trixi.jl.
The default is `"sqrt_Trixi_NaN"` which returns `NaN` for negative arguments
instead of throwing an error.
Alternatively, you can set `type` to `"sqrt_Base"` to use the Julia built-in `sqrt` function
which provides a stack-trace of the error which might come in handy when debugging code.
"""
function set_sqrt_type!(type; force = true)
    @assert type == "sqrt_Trixi_NaN"||type == "sqrt_Base" "Only allowed `sqrt` function types are `\"sqrt_Trixi_NaN\"` and `\"sqrt_Base\"`"
    set_preferences!(TRIXI_UUID, "sqrt" => type, force = force)
    @info "Please restart Julia and reload Trixi.jl for the `sqrt` computation change to take effect"
end

# TODO: deprecation introduced in v0.8
@deprecate set_sqrt_type(type; force = true) set_sqrt_type!(type; force = true) false

@static if _PREFERENCE_SQRT == "sqrt_Trixi_NaN"
    """
        Trixi.sqrt(x::Real)

    Custom square root function which returns `NaN` for negative arguments instead of throwing an error.
    This is required to ensure [correct results for multithreaded computations](https://github.com/trixi-framework/Trixi.jl/issues/1766)
    when using the [`Polyester` package](https://github.com/JuliaSIMD/Polyester.jl),
    i.e., using the `@batch` macro instead of the Julia built-in `@threads` macro, see [`@threaded`](@ref).

    We dispatch this function for `Float64, Float32, Float16` to the LLVM intrinsics
    `llvm.sqrt.f64`, `llvm.sqrt.f32`, `llvm.sqrt.f16` as for these the LLVM functions can be used out-of the box,
    i.e., they return `NaN` for negative arguments.
    In principle, one could also use the `sqrt_llvm` call, but for transparency and consistency with [`log`](@ref) we
    spell out the datatype-dependent functions here.
    For other types, such as integers or dual numbers required for algorithmic differentiation, we
    fall back to the Julia built-in `sqrt` function after a check for negative arguments.
    Since these cases are not performance critical, the check for negativity does not hurt here
    and can (as of now) even be optimized away by the compiler due to the implementation of `sqrt` in Julia.

    When debugging code, it might be useful to change the implementation of this function to redirect to
    the Julia built-in `sqrt` function, as this reports the exact place in code where the domain is violated
    in the stacktrace.

    See also [`Trixi.set_sqrt_type!`](@ref).
    """
    @inline sqrt(x::Real) = x < zero(x) ? oftype(x, NaN) : Base.sqrt(x)

    # For `sqrt` we could use the `sqrt_llvm` call, ...
    #@inline sqrt(x::Union{Float64, Float32, Float16}) = Base.sqrt_llvm(x)

    # ... but for transparency and consistency we use the direct LLVM calls here.
    @inline sqrt(x::Float64) = ccall("llvm.sqrt.f64", llvmcall, Float64, (Float64,), x)
    @inline sqrt(x::Float32) = ccall("llvm.sqrt.f32", llvmcall, Float32, (Float32,), x)
    @inline sqrt(x::Float16) = ccall("llvm.sqrt.f16", llvmcall, Float16, (Float16,), x)
end

"""
    Trixi.set_log_type!(type; force = true)

Set the `type` of the (natural) `log` function to be used in Trixi.jl.
The default is `"sqrt_Trixi_NaN"` which returns `NaN` for negative arguments
instead of throwing an error.
Alternatively, you can set `type` to `"sqrt_Base"` to use the Julia built-in `sqrt` function
which provides a stack-trace of the error which might come in handy when debugging code.
"""
function set_log_type!(type; force = true)
    @assert type == "log_Trixi_NaN"||type == "log_Base" "Only allowed log function types are `\"log_Trixi_NaN\"` and `\"log_Base\"`."
    set_preferences!(TRIXI_UUID, "log" => type, force = force)
    @info "Please restart Julia and reload Trixi.jl for the `log` computation change to take effect"
end

# TODO: deprecation introduced in v0.8
@deprecate set_log_type(type; force = true) set_log_type!(type; force = true) false

@static if _PREFERENCE_LOG == "log_Trixi_NaN"
    """
        Trixi.log(x::Real)

    Custom natural logarithm function which returns `NaN` for negative arguments instead of throwing an error.
    This is required to ensure [correct results for multithreaded computations](https://github.com/trixi-framework/Trixi.jl/issues/1766)
    when using the [`Polyester` package](https://github.com/JuliaSIMD/Polyester.jl),
    i.e., using the `@batch` macro instead of the Julia built-in `@threads` macro, see [`@threaded`](@ref).

    We dispatch this function for `Float64, Float32, Float16` to the respective LLVM intrinsics
    `llvm.log.f64`, `llvm.log.f32`, `llvm.log.f16` as for this the LLVM functions can be used out-of the box, i.e.,
    they return `NaN` for negative arguments.
    For other types, such as integers or dual numbers required for algorithmic differentiation, we
    fall back to the Julia built-in `log` function after a check for negative arguments.
    Since these cases are not performance critical, the check for negativity does not hurt here.

    When debugging code, it might be useful to change the implementation of this function to redirect to
    the Julia built-in `log` function, as this reports the exact place in code where the domain is violated
    in the stacktrace.

    See also [`Trixi.set_log_type!`](@ref).
    """
    @inline log(x::Real) = x < zero(x) ? oftype(x, NaN) : Base.log(x)

    @inline log(x::Float64) = ccall("llvm.log.f64", llvmcall, Float64, (Float64,), x)
    @inline log(x::Float32) = ccall("llvm.log.f32", llvmcall, Float32, (Float32,), x)
    @inline log(x::Float16) = ccall("llvm.log.f16", llvmcall, Float16, (Float16,), x)
end

"""
    Trixi.ln_mean(x::Real, y::Real)

Compute the logarithmic mean

    ln_mean(x, y) = (y - x) / (log(y) - log(x)) = (y - x) / log(y / x)

Problem: The formula above has a removable singularity at `x == y`. Thus,
some care must be taken to implement it correctly without problems or loss
of accuracy when `x ≈ y`. Here, we use the approach proposed by
Ismail and Roe (2009).
Set ξ = y / x. Then, we have

    (y - x) / log(y / x) = (x + y) / log(ξ) * (ξ - 1) / (ξ + 1)

Set f = (ξ - 1) / (ξ + 1) = (y - x) / (x + y). Then, we use the expansion

    log(ξ) = 2 * f * (1 + f^2 / 3 + f^4 / 5 + f^6 / 7) + O(ξ^9)

Inserting the first few terms of this expansion yields

    (y - x) / log(ξ) ≈ (x + y) * f / (2 * f * (1 + f^2 / 3 + f^4 / 5 + f^6 / 7))
                     = (x + y) / (2 + 2/3 * f^2 + 2/5 * f^4 + 2/7 * f^6)

Since divisions are usually more expensive on modern hardware than
multiplications (Agner Fog), we try to avoid computing two divisions. Thus,
we use

    f^2 = (y - x)^2 / (x + y)^2
        = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)

Given ε = 1.0e-4, we use the following algorithm.

    if f^2 < ε
      # use the expansion above
    else
      # use the direct formula (y - x) / log(y / x)
    end

# References
- Ismail, Roe (2009).
  Affordable, entropy-consistent Euler flux functions II: Entropy production at shocks.
  [DOI: 10.1016/j.jcp.2009.04.021](https://doi.org/10.1016/j.jcp.2009.04.021)
- Agner Fog.
  Lists of instruction latencies, throughputs and micro-operation breakdowns
  for Intel, AMD, and VIA CPUs.
  [https://www.agner.org/optimize/instruction_tables.pdf](https://www.agner.org/optimize/instruction_tables.pdf)
"""
@inline ln_mean(x::Real, y::Real) = ln_mean(promote(x, y)...)

@inline function ln_mean(x::RealT, y::RealT) where {RealT <: Real}
    epsilon_f2 = convert(RealT, 1.0e-4)
    f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y) # f2 = f^2
    if f2 < epsilon_f2
        return (x + y) / @evalpoly(f2,
                         2,
                         convert(RealT, 2 / 3),
                         convert(RealT, 2 / 5),
                         convert(RealT, 2 / 7))
    else
        return (y - x) / log(y / x)
    end
end

"""
    Trixi.inv_ln_mean(x::Real, y::Real)

Compute the inverse `1 / ln_mean(x, y)` of the logarithmic mean
[`ln_mean`](@ref).

This function may be used to increase performance where the inverse of the
logarithmic mean is needed, by replacing a (slow) division by a (fast)
multiplication.
"""
@inline inv_ln_mean(x::Real, y::Real) = inv_ln_mean(promote(x, y)...)

@inline function inv_ln_mean(x::RealT, y::RealT) where {RealT <: Real}
    epsilon_f2 = convert(RealT, 1.0e-4)
    f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y) # f2 = f^2
    if f2 < epsilon_f2
        return @evalpoly(f2,
                         2,
                         convert(RealT, 2 / 3),
                         convert(RealT, 2 / 5),
                         convert(RealT, 2 / 7)) / (x + y)
    else
        return log(y / x) / (y - x)
    end
end

# `Base.max` and `Base.min` perform additional checks for signed zeros and `NaN`s
# which are not present in comparable functions in Fortran/C++. For example,
# ```julia
# julia> @code_native debuginfo=:none syntax=:intel max(1.0, 2.0)
#         .text
#         vmovq   rcx, xmm1
#         vmovq   rax, xmm0
#         vcmpordsd       xmm2, xmm0, xmm0
#         vblendvpd       xmm2, xmm0, xmm1, xmm2
#         vcmpordsd       xmm3, xmm1, xmm1
#         vblendvpd       xmm3, xmm1, xmm0, xmm3
#         vmovapd xmm4, xmm2
#         test    rcx, rcx
#         jns     L45
#         vmovapd xmm4, xmm3
# L45:
#         test    rax, rax
#         js      L54
#         vmovapd xmm4, xmm3
# L54:
#         vcmpltsd        xmm0, xmm0, xmm1
#         vblendvpd       xmm0, xmm4, xmm2, xmm0
#         ret
#         nop     word ptr cs:[rax + rax]
#
# julia> @code_native debuginfo=:none syntax=:intel min(1.0, 2.0)
#         .text
#         vmovq   rcx, xmm1
#         vmovq   rax, xmm0
#         vcmpordsd       xmm2, xmm0, xmm0
#         vblendvpd       xmm2, xmm0, xmm1, xmm2
#         vcmpordsd       xmm3, xmm1, xmm1
#         vblendvpd       xmm3, xmm1, xmm0, xmm3
#         vmovapd xmm4, xmm2
#         test    rcx, rcx
#         jns     L58
#         test    rax, rax
#         js      L67
# L46:
#         vcmpltsd        xmm0, xmm1, xmm0
#         vblendvpd       xmm0, xmm4, xmm2, xmm0
#         ret
# L58:
#         vmovapd xmm4, xmm3
#         test    rax, rax
#         jns     L46
# L67:
#         vmovapd xmm4, xmm3
#         vcmpltsd        xmm0, xmm1, xmm0
#         vblendvpd       xmm0, xmm4, xmm2, xmm0
#         ret
#         nop     word ptr cs:[rax + rax]
#         nop     dword ptr [rax]
# ```
# In contrast, we get the much simpler and faster version
# ```julia
# julia> @code_native debuginfo=:none syntax=:intel Base.FastMath.max_fast(1.0, 2.0)
#         .text
#         vmaxsd  xmm0, xmm1, xmm0
#         ret
#         nop     word ptr cs:[rax + rax]
#
# julia> @code_native debuginfo=:none syntax=:intel Base.FastMath.min_fast(1.0, 2.0)
#         .text
#         vminsd  xmm0, xmm0, xmm1
#         ret
#         nop     word ptr cs:[rax + rax]
# ```
# when using `@fastmath`, which we also get from
# [Fortran](https://godbolt.org/z/Yrsa1js7P)
# or [C++](https://godbolt.org/z/674G7Pccv).
#
# Note however that such a custom reimplementation can cause incompatibilities with other
# packages. Currently we are affected by an issue with MPI.jl on ARM, see
# https://github.com/trixi-framework/Trixi.jl/issues/1922
# The workaround is to resort to Base.min / Base.max when using MPI reductions.
"""
    Trixi.max(x, y, ...)

Return the maximum of the arguments. See also the `maximum` function to take
the maximum element from a collection.

This version in Trixi.jl is semantically equivalent to `Base.max` but may
be implemented differently. In particular, it may avoid potentially expensive
checks necessary in the presence of `NaN`s (or signed zeros).

# Examples

```jldoctest
julia> max(2, 5, 1)
5
```
"""
@inline max(args...) = @fastmath max(args...)

"""
    Trixi.min(x, y, ...)

Return the minimum of the arguments. See also the `minimum` function to take
the minimum element from a collection.

This version in Trixi.jl is semantically equivalent to `Base.min` but may
be implemented differently. In particular, it may avoid potentially expensive
checks necessary in the presence of `NaN`s (or signed zeros).

# Examples

```jldoctest
julia> min(2, 5, 1)
1
```
"""
@inline min(args...) = @fastmath min(args...)

"""
    Trixi.positive_part(x)

Return `x` if `x` is positive, else zero. In other words, return
`(x + abs(x)) / 2` for real numbers `x`.
"""
@inline function positive_part(x)
    return max(x, zero(x))
end

"""
    Trixi.negative_part(x)

Return `x` if `x` is negative, else zero. In other words, return
`(x - abs(x)) / 2` for real numbers `x`.
"""
@inline function negative_part(x)
    return min(x, zero(x))
end

"""
    Trixi.stolarsky_mean(x::Real, y:Real, gamma::Real)

Compute an instance of a weighted Stolarsky mean of the form

    stolarsky_mean(x, y, gamma) = (gamma - 1)/gamma * (y^gamma - x^gamma) / (y^(gamma-1) - x^(gamma-1))

where `gamma > 1`.

Problem: The formula above has a removable singularity at `x == y`. Thus,
some care must be taken to implement it correctly without problems or loss
of accuracy when `x ≈ y`. Here, we use the approach proposed by
Winters et al. (2020).
Set f = (y - x) / (y + x) and g = gamma (for compact notation).
Then, we use the expansions

    ((1+f)^g - (1-f)^g) / g = 2*f + (g-1)(g-2)/3 * f^3 + (g-1)(g-2)(g-3)(g-4)/60 * f^5 + O(f^7)

and

    ((1+f)^(g-1) - (1-f)^(g-1)) / (g-1) = 2*f + (g-2)(g-3)/3 * f^3 + (g-2)(g-3)(g-4)(g-5)/60 * f^5 + O(f^7)

Inserting the first few terms of these expansions and performing polynomial long division
we find that

    stolarsky_mean(x, y, gamma) ≈ (y + x) / 2 * (1 + (g-2)/3 * f^2 - (g+1)(g-2)(g-3)/45 * f^4 + (g+1)(g-2)(g-3)(2g(g-2)-9)/945 * f^6)

Since divisions are usually more expensive on modern hardware than
multiplications (Agner Fog), we try to avoid computing two divisions. Thus,
we use

    f^2 = (y - x)^2 / (x + y)^2
        = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y)

Given ε = 1.0e-4, we use the following algorithm.

    if f^2 < ε
      # use the expansion above
    else
      # use the direct formula (gamma - 1)/gamma * (y^gamma - x^gamma) / (y^(gamma-1) - x^(gamma-1))
    end

# References
- Andrew R. Winters, Christof Czernik, Moritz B. Schily & Gregor J. Gassner (2020)
  Entropy stable numerical approximations for the isothermal and polytropic
  Euler equations
  [DOI: 10.1007/s10543-019-00789-w](https://doi.org/10.1007/s10543-019-00789-w)
- Agner Fog.
  Lists of instruction latencies, throughputs and micro-operation breakdowns
  for Intel, AMD, and VIA CPUs.
  [https://www.agner.org/optimize/instruction_tables.pdf](https://www.agner.org/optimize/instruction_tables.pdf)
"""
@inline stolarsky_mean(x::Real, y::Real, gamma::Real) = stolarsky_mean(promote(x, y,
                                                                               gamma)...)

@inline function stolarsky_mean(x::RealT, y::RealT, gamma::RealT) where {RealT <: Real}
    epsilon_f2 = convert(RealT, 1.0e-4)
    f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y) # f2 = f^2
    if f2 < epsilon_f2
        # convenience coefficients
        c1 = convert(RealT, 1 / 3) * (gamma - 2)
        c2 = convert(RealT, -1 / 15) * (gamma + 1) * (gamma - 3) * c1
        c3 = convert(RealT, -1 / 21) * (2 * gamma * (gamma - 2) - 9) * c2
        return 0.5f0 * (x + y) * @evalpoly(f2, 1, c1, c2, c3)
    else
        return (gamma - 1) / gamma * (y^gamma - x^gamma) /
               (y^(gamma - 1) - x^(gamma - 1))
    end
end
end # @muladd
