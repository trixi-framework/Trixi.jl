# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Calculate correction matrices for the Flux Reconstruction (FR) method,
# see `SurfaceIntegralFluxReconstruction`. 

@doc raw"""
    correction_function_DG()

Returns `(Val(:g_DG),)` representing the "Lumped Lobatto" correction function ``g_\mathrm{DG}``.
Must be supplied to [`SurfaceIntegralFluxReconstruction`](@ref) as the
`correction_function` keyword argument.

## Reference

- Huynh (2007)
  "A Flux Reconstruction Approach to High-Order Schemes Including Discontinuous Galerkin Methods"
  [DOI: 10.2514/6.2007-4079](https://doi.org/10.2514/6.2007-4079)
"""
correction_function_DG() = (Val(:g_DG),)

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
    correction_function_2()

Returns `(Val(:g_2),)` representing the "Lumped Lobatto" correction function ``g_2``.
Must be supplied to [`SurfaceIntegralFluxReconstruction`](@ref) as the
`correction_function` keyword argument.

## Reference

- Huynh (2007)
  "A Flux Reconstruction Approach to High-Order Schemes Including Discontinuous Galerkin Methods"
  [DOI: 10.2514/6.2007-4079](https://doi.org/10.2514/6.2007-4079)
"""
correction_function_2() = (Val(:g_2),)

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
end # @muladd
