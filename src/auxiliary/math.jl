# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    ln_mean(x, y)

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
@inline function ln_mean(x, y)
    epsilon_f2 = 1.0e-4
    f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y) # f2 = f^2
    if f2 < epsilon_f2
        return (x + y) / @evalpoly(f2, 2, 2/3, 2/5, 2/7)
    else
        return (y - x) / log(y / x)
    end
end

"""
    inv_ln_mean(x, y)

Compute the inverse `1 / ln_mean(x, y)` of the logarithmic mean
[`ln_mean`](@ref).

This function may be used to increase performance where the inverse of the
logarithmic mean is needed, by replacing a (slow) division by a (fast)
multiplication.
"""
@inline function inv_ln_mean(x, y)
    epsilon_f2 = 1.0e-4
    f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y) # f2 = f^2
    if f2 < epsilon_f2
        return @evalpoly(f2, 2, 2/3, 2/5, 2/7) / (x + y)
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
"""
    max(x, y, ...)

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
    min(x, y, ...)

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
    positive_part(x)

Return `x` if `x` is positive, else zero. In other words, return
`(x + abs(x)) / 2` for real numbers `x`.
"""
@inline function positive_part(x)
    return max(x, zero(x))
end

"""
    negative_part(x)

Return `x` if `x` is negative, else zero. In other words, return
`(x - abs(x)) / 2` for real numbers `x`.
"""
@inline function negative_part(x)
    return min(x, zero(x))
end

"""
    stolarsky_mean(x, y, gamma)

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
@inline function stolarsky_mean(x, y, gamma)
    epsilon_f2 = 1.0e-4
    f2 = (x * (x - 2 * y) + y * y) / (x * (x + 2 * y) + y * y) # f2 = f^2
    if f2 < epsilon_f2
        # convenience coefficients
        c1 = (1 / 3) * (gamma - 2)
        c2 = -(1 / 15) * (gamma + 1) * (gamma - 3) * c1
        c3 = -(1 / 21) * (2 * gamma * (gamma - 2) - 9) * c2
        return 0.5 * (x + y) * @evalpoly(f2, 1, c1, c2, c3)
    else
        return (gamma - 1) / gamma * (y^gamma - x^gamma) /
               (y^(gamma - 1) - x^(gamma - 1))
    end
end
end # @muladd
