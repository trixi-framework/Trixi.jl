# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Calculate correction matrices for the Flux Reconstruction (FR) method,
# see `SurfaceIntegralFluxReconstruction`. 

@doc raw"""
    correction_function_DG(c)

Returns `Val(:g_DG)` representing the "Lumped Lobatto" correction function ```g_\mathrm{DG}``.
Must be supplied to [`SurfaceIntegralFluxReconstruction`](@ref) as the
`correction_function` keyword argument.

## Reference

- Huynh (2007)
  "A Flux Reconstruction Approach to High-Order Schemes Including Discontinuous Galerkin Methods"
  [DOI: 10.2514/6.2007-4079](https://doi.org/10.2514/6.2007-4079)
"""
correction_function_DG() = (Val(:g_DG),)

# Implements Huynh's `g_DG` correction function, see
# - Huynh (2007)
#  "A Flux Reconstruction Approach to High-Order Schemes IncludingDiscontinuous Galerkin Methods"
#  [DOI: 10.2514/6.2007-4079](https://doi.org/10.2514/6.2007-4079)
function calc_correction_matrix(basis, ::Val{:g_DG})
    nodes = basis.nodes
    RealT = eltype(nodes)
    K = nnodes(basis) # notation from Huynh (2007)
    g_derivative_matrix = zeros(RealT, K, 2)

    for i in 1:K
        xi = nodes[i]
        # `legendre_polynomial_and_derivative` returns "normalized" (multiplied by sqrt(K + 0.5)) 
        # polynomial and derivative
        _, dL_Km1_normalized = legendre_polynomial_and_derivative(K - 1, xi)
        _, dL_K_normalized = legendre_polynomial_and_derivative(K, xi)

        # Undo the normalization to get standard Legendre polynomial derivatives
        dL_Km1 = dL_Km1_normalized / sqrt(K - 1 + 0.5)
        dL_K = dL_K_normalized / sqrt(K + 0.5)

        # Use right Radau polynomial R_R (1 at -1, 0 at 1).
        # See eq. (4.1) for left correction function:
        dR_RK = (-1)^K / 2 * (dL_K - dL_Km1) # right Radau polynomial degree K
        g_derivative_matrix[i, 1] = dR_RK

        # Use left Radau polynomial R_L (0 at -1, 1 at 1).
        # See eq. (A.17) for right correction function:
        dR_LK = 0.5 * (dL_Km1 + dL_K) # left Radau polynomial degree K
        g_derivative_matrix[i, 2] = dR_LK
    end

    return g_derivative_matrix
end

@doc raw"""
    correction_function_2(c)

Returns `Val(:g_2)` representing the "Lumped Lobatto" correction function ``g_2``.
Must be supplied to [`SurfaceIntegralFluxReconstruction`](@ref) as the
`correction_function` keyword argument.

## Reference

- Huynh (2007)
  "A Flux Reconstruction Approach to High-Order Schemes Including Discontinuous Galerkin Methods"
  [DOI: 10.2514/6.2007-4079](https://doi.org/10.2514/6.2007-4079)
"""
correction_function_2() = (Val(:g_2),)

# Implements Huynh's `g_2` correction function, see
# - Huynh (2007)
#  "A Flux Reconstruction Approach to High-Order Schemes IncludingDiscontinuous Galerkin Methods"
#  [DOI: 10.2514/6.2007-4079](https://doi.org/10.2514/6.2007-4079)
function calc_correction_matrix(basis, ::Val{:g_2})
    nodes = basis.nodes
    RealT = eltype(nodes)
    K = nnodes(basis) # notation from Huynh (2007)
    g_derivative_matrix = zeros(RealT, K, 2)

    for i in 1:K
        xi = nodes[i]
        # `legendre_polynomial_and_derivative` returns "normalized" (multiplied by sqrt(K + 0.5)) 
        # polynomial and derivative
        _, dL_Km2_normalized = legendre_polynomial_and_derivative(K - 2, xi)
        _, dL_Km1_normalized = legendre_polynomial_and_derivative(K - 1, xi)
        _, dL_K_normalized = legendre_polynomial_and_derivative(K, xi)

        # Undo the normalization to get standard Legendre polynomial derivatives
        dL_Km2 = dL_Km2_normalized / sqrt(K - 2 + 0.5)
        dL_Km1 = dL_Km1_normalized / sqrt(K - 1 + 0.5)
        dL_K = dL_K_normalized / sqrt(K + 0.5)

        # Build derivatives of right Radau polynomials R_R
        dR_RKm1 = (-1)^(K - 1) / 2 * (dL_Km1 - dL_Km2) # right Radau polynomial degree K-1
        dR_RK = (-1)^K / 2 * (dL_K - dL_Km1) # right Radau polynomial degree K

        # Use "right" Radau polynomial R_R (1 at -1, 0 at 1).
        # See eq. (4.4) for left correction function g_2 = g_{2, L}:
        g_derivative_matrix[i, 1] = ((K - 1) * dR_RK + K * dR_RKm1) / (2 * K - 1)

        # Build derivatives of left Radau polynomials R_L
        dR_LKm1 = 0.5 * (dL_Km2 + dL_Km1) # left Radau polynomial degree K-1
        dR_LK = 0.5 * (dL_Km1 + dL_K) # left Radau polynomial degree K

        # Use "left" Radau polynomial R_L (0 at -1, 1 at 1).
        # This is eq. (4.4) with left instead of right Radau polynomials:
        g_derivative_matrix[i, 2] = ((K - 1) * dR_LK - K * dR_LKm1) / (2 * K - 1)
    end

    return g_derivative_matrix
end

@doc raw"""
    correction_function_ESFR(c)

Returns a `(Val(:g_ESFR), c)` representing the Energy Stable Flux Reconstruction (ESFR)
correction function with parameter `c`.
Must be supplied to [`SurfaceIntegralFluxReconstruction`](@ref) as the
`correction_function` keyword argument.

Choices for c are :
- `c > c_min_ESFR(k)`: Any value greater than the minimum value for stability, see [`c_min_ESFR`](@ref).
- `0`: Classic strong form DG correction function, recovers [`correction_function_DG`](@ref)
- [`c_SD(k)`](@ref): Spectral Difference correction function
- [`c_HU(k)`](@ref): Recovers Huynh's Lumped Lobatto correction function ``g_2``, see also [`correction_function_2`](@ref)

## Reference

- Vincent, Castonguay, Jameson (2011)
  "A New Class of High-Order Energy Stable Flux Reconstruction Schemes"
  [DOI: 10.1007/s10915-010-9420-z](https://doi.org/10.1007/s10915-010-9420-z)
"""
correction_function_ESFR(c) = (Val(:g_ESFR), c)

#a_k(k) = factorial(2 * k) / (2^k * (factorial(k))^2) # eq. (3.24)
a_k_mod(k) = factorial(2 * k) / (2^k * factorial(k)) # more stable version

"""
    c_min_ESFR(k)

Returns the minimum value of the parameter `c` for stability of the
Energy Stable Flux Reconstruction (ESFR) correction function for polynomial degree `k`.
Must be supplied to [`correction_function_ESFR`](@ref).

## Reference

- Vincent, Castonguay, Jameson (2011)
  "A New Class of High-Order Energy Stable Flux Reconstruction Schemes"
  [DOI: 10.1007/s10915-010-9420-z](https://doi.org/10.1007/s10915-010-9420-z)
"""
function c_min_ESFR(k)
    #return -2 / ((2 * k + 1) * (a_k(k) * factorial(k))^2) # eq. (3.28)
    return -2 / ((2 * k + 1) * a_k_mod(k)^2) # more stable version
end

"""
    c_SD(k)

Returns the value of the parameter `c` for the Spectral Difference (SD) correction function
for polynomial degree `k`.
Must be supplied to [`correction_function_ESFR`](@ref).

## Reference

- Vincent, Castonguay, Jameson (2011)
  "A New Class of High-Order Energy Stable Flux Reconstruction Schemes"
  [DOI: 10.1007/s10915-010-9420-z](https://doi.org/10.1007/s10915-010-9420-z)
"""
function c_SD(k)
    #return (2 * k) / ( (2 * k + 1) * (k + 1) * (a_k(k) * factorial(k))^2 ) # eq. (3.50)
    return (2 * k) / ((2 * k + 1) * (k + 1) * a_k_mod(k)^2) # more stable version
end

"""
    c_HU(k)

Returns the value of the parameter `c` for Huynh's `g_2` correction function
for polynomial degree `k`.
Must be supplied to [`correction_function_ESFR`](@ref).
"""
function c_HU(k)
    #return (2 * (k + 1))/((2 * k + 1) * k * (a_k(k) * factorial(k))^2) # eq. (3.54
    return (2 * (k + 1)) / ((2 * k + 1) * k * a_k_mod(k)^2) # more stable version
end

# The ESFR schemes form a one-parameter family controlled by the parameter `c`.
# Special cases:
# - c > c_min_ESFR: Lower bound for stability
# - c = 0: recovers the DG scheme (equivalent to g_DG)
# - c = c_SD: recovers the Spectral Difference (SD) scheme
# - c = c_HU: recovers Huynh's g_2 scheme (equivalent to g_2)
function calc_correction_matrix(basis, ::Val{:g_ESFR}, c)
    nodes = basis.nodes
    RealT = eltype(nodes)
    N = nnodes(basis)
    k = N - 1 # notation from Vincent et al. (2011)
    g_derivative_matrix = zeros(RealT, N, 2)

    #eta_k = c * (2 * k + 1) * (a_k(k) * factorial(k))^2 # eq. (3.45)
    #eta_k = c * (2 * k + 1) * a_k_mod(k)^2 # more stable version

    for i in 1:N
        xi = nodes[i]
        # `legendre_polynomial_and_derivative` returns "normalized" (multiplied by sqrt(K + 0.5)) 
        # polynomial and derivative
        _, dL_km1_normalized = legendre_polynomial_and_derivative(k - 1, xi)
        _, dL_k_normalized = legendre_polynomial_and_derivative(k, xi)
        _, dL_kp1_normalized = legendre_polynomial_and_derivative(k + 1, xi)

        # Undo the normalization to get standard Legendre polynomial derivatives
        dL_km1 = dL_km1_normalized / sqrt(k - 1 + 0.5)
        dL_k = dL_k_normalized / sqrt(k + 0.5)
        dL_kp1 = dL_kp1_normalized / sqrt(k + 1 + 0.5)

        # Common term in left and right correction function derivatives
        eta_term = (eta_k * dL_km1 + dL_kp1) / (1 + eta_k)

        # Left correction function derivative (at ξ = -1 boundary)
        g_derivative_matrix[i, 1] = (-1)^k / 2 * (dL_k - eta_term) # eq. (3.46)

        # Right correction function derivative (at ξ = +1 boundary)
        g_derivative_matrix[i, 2] = 0.5 * (dL_k + eta_term) # eq. (3.47)
    end

    return g_derivative_matrix
end
end # @muladd
