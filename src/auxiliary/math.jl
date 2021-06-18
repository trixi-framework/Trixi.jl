
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
  https://www.agner.org/optimize/instruction_tables.pdf
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
