# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This diagram shows what is meant by "lower", "upper", and "large":
#      +1   +1
#       |    |
# upper |    |
#       |    |
#      -1    |
#            | large
#      +1    |
#       |    |
# lower |    |
#       |    |
#      -1   -1
#
# That is, we are only concerned with 2:1 subdivision of a surface/element.

# Calculate forward projection matrix for discrete L2 projection from large to upper
#
# Note: This is actually an interpolation.
function calc_forward_upper(n_nodes, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights
    nodes, _ = gauss_lobatto_nodes_weights(n_nodes, RealT)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: interpolation)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (nodes[j] + 1), nodes, wbary)
        for i in 1:n_nodes
            operator[j, i] = poly[i]
        end
    end

    return operator
end

# Calculate forward projection matrix for discrete L2 projection from large to lower
#
# Note: This is actually an interpolation.
function calc_forward_lower(n_nodes, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights
    nodes, _ = gauss_lobatto_nodes_weights(n_nodes, RealT)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: interpolation)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (nodes[j] - 1), nodes, wbary)
        for i in 1:n_nodes
            operator[j, i] = poly[i]
        end
    end

    return operator
end

# Calculate reverse projection matrix for discrete L2 projection from upper to large (Gauss version)
#
# Note: To not make the L2 projection exact, first convert to Gauss nodes,
# perform projection, and convert back to Gauss-Lobatto.
function calc_reverse_upper(n_nodes, ::Val{:gauss}, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights for Legendre-Gauss
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes, RealT)
    gauss_wbary = barycentric_weights(gauss_nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (gauss_nodes[j] + 1),
                                                  gauss_nodes, gauss_wbary)
        for i in 1:n_nodes
            operator[i, j] = 0.5f0 * poly[i] * gauss_weights[j] / gauss_weights[i]
        end
    end

    # Calculate Vandermondes for quadrature-interpolation basis transform
    lobatto_nodes, _ = gauss_lobatto_nodes_weights(n_nodes, RealT)
    gauss2lobatto = polynomial_interpolation_matrix(gauss_nodes, lobatto_nodes)
    lobatto2gauss = polynomial_interpolation_matrix(lobatto_nodes, gauss_nodes)

    return gauss2lobatto * operator * lobatto2gauss
end

# Calculate reverse projection matrix for discrete L2 projection from lower to large (Gauss version)
#
# Note: To not make the L2 projection exact, first convert to Gauss nodes,
# perform projection, and convert back to Gauss-Lobatto.
function calc_reverse_lower(n_nodes, ::Val{:gauss}, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights for Legendre-Gauss
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes, RealT)
    gauss_wbary = barycentric_weights(gauss_nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (gauss_nodes[j] - 1),
                                                  gauss_nodes, gauss_wbary)
        for i in 1:n_nodes
            operator[i, j] = 0.5f0 * poly[i] * gauss_weights[j] / gauss_weights[i]
        end
    end

    # Calculate Vandermondes for quadrature-interpolation basis transform
    lobatto_nodes, _ = gauss_lobatto_nodes_weights(n_nodes, RealT)
    gauss2lobatto = polynomial_interpolation_matrix(gauss_nodes, lobatto_nodes)
    lobatto2gauss = polynomial_interpolation_matrix(lobatto_nodes, gauss_nodes)

    return gauss2lobatto * operator * lobatto2gauss
end

# Calculate reverse projection matrix for discrete L2 projection from upper to large (Gauss-Lobatto
# version)
function calc_reverse_upper(n_nodes, ::Val{:gauss_lobatto}, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights
    nodes, weights = gauss_lobatto_nodes_weights(n_nodes, RealT)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (nodes[j] + 1), nodes, wbary)
        for i in 1:n_nodes
            operator[i, j] = 0.5f0 * poly[i] * weights[j] / weights[i]
        end
    end

    return operator
end

# Calculate reverse projection matrix for discrete L2 projection from lower to large (Gauss-Lobatto
# version)
function calc_reverse_lower(n_nodes, ::Val{:gauss_lobatto}, RealT = Float64)
    # Calculate nodes, weights, and barycentric weights
    nodes, weights = gauss_lobatto_nodes_weights(n_nodes, RealT)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(RealT, n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(0.5f0 * (nodes[j] - 1), nodes, wbary)
        for i in 1:n_nodes
            operator[i, j] = 0.5f0 * poly[i] * weights[j] / weights[i]
        end
    end

    return operator
end

# Compute the L2 projection matrix for projecting polynomials 
# from a higher degree to a lower degree using Gauss-Legendre quadrature.
#
# Arguments
# - `nodes_high`: GLL/LGL nodes of the higher-degree polynomial
# - `nodes_low`: GLL/LGL nodes of the lower-degree polynomial
# - `::Val{:gauss}`: Use Gauss-Legendre quadrature (accuracy 2N - 1)
# - `RealT`: Type of the output matrix (default: Float64)
#
# Returns
# The projection matrix such that multiplying with it projects 
# a higher degree Lagrange interpolation/solution polynomial
# to a lower degree Lagrange interpolation/solution polynomial.
function polynomial_l2projection_matrix(nodes_high, nodes_low, ::Val{:gauss},
                                        RealT = Float64)
    n_high = length(nodes_high)
    n_low = length(nodes_low)

    # Get Gauss-Legendre nodes and weights for quadrature
    # Use enough nodes to exactly integrate polynomials of degree n_high + n_low - 1
    n_quad = div(n_high + n_low + 1, 2)
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_quad, RealT)

    # Get barycentric weights for interpolation
    wbary_high = barycentric_weights(nodes_high)
    wbary_low = barycentric_weights(nodes_low)

    # Weights for the low-degree mass matrix (diagonal for Gauss-Lobatto)
    weights_low = gauss_lobatto_nodes_weights(n_low, RealT)[2]

    # Build projection matrix
    projection_matrix = zeros(RealT, n_low, n_high)

    for q in 1:n_quad
        # Evaluate low-degree basis functions at Gauss quadrature point
        poly_low = lagrange_interpolating_polynomials(gauss_nodes[q], nodes_low,
                                                      wbary_low)

        # Evaluate high-degree basis functions at Gauss quadrature point
        poly_high = lagrange_interpolating_polynomials(gauss_nodes[q], nodes_high,
                                                       wbary_high)

        for i in 1:n_low, j in 1:n_high
            # Build integral using Gauss quadrature
            projection_matrix[i, j] += poly_low[i] * poly_high[j] * gauss_weights[q] /
                                       weights_low[i]
        end
    end

    return projection_matrix
end
end # @muladd
