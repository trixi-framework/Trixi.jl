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
function calc_forward_upper(n_nodes)
    # Calculate nodes, weights, and barycentric weights
    nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: interpolation)
    operator = zeros(n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(1 / 2 * (nodes[j] + 1), nodes, wbary)
        for i in 1:n_nodes
            operator[j, i] = poly[i]
        end
    end

    return operator
end

# Calculate forward projection matrix for discrete L2 projection from large to lower
#
# Note: This is actually an interpolation.
function calc_forward_lower(n_nodes)
    # Calculate nodes, weights, and barycentric weights
    nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: interpolation)
    operator = zeros(n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(1 / 2 * (nodes[j] - 1), nodes, wbary)
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
function calc_reverse_upper(n_nodes, ::Val{:gauss})
    # Calculate nodes, weights, and barycentric weights for Legendre-Gauss
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes)
    gauss_wbary = barycentric_weights(gauss_nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(1 / 2 * (gauss_nodes[j] + 1),
                                                  gauss_nodes, gauss_wbary)
        for i in 1:n_nodes
            operator[i, j] = 1 / 2 * poly[i] * gauss_weights[j] / gauss_weights[i]
        end
    end

    # Calculate Vandermondes
    lobatto_nodes, lobatto_weights = gauss_lobatto_nodes_weights(n_nodes)
    gauss2lobatto = polynomial_interpolation_matrix(gauss_nodes, lobatto_nodes)
    lobatto2gauss = polynomial_interpolation_matrix(lobatto_nodes, gauss_nodes)

    return gauss2lobatto * operator * lobatto2gauss
end

# Calculate reverse projection matrix for discrete L2 projection from lower to large (Gauss version)
#
# Note: To not make the L2 projection exact, first convert to Gauss nodes,
# perform projection, and convert back to Gauss-Lobatto.
function calc_reverse_lower(n_nodes, ::Val{:gauss})
    # Calculate nodes, weights, and barycentric weights for Legendre-Gauss
    gauss_nodes, gauss_weights = gauss_nodes_weights(n_nodes)
    gauss_wbary = barycentric_weights(gauss_nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(1 / 2 * (gauss_nodes[j] - 1),
                                                  gauss_nodes, gauss_wbary)
        for i in 1:n_nodes
            operator[i, j] = 1 / 2 * poly[i] * gauss_weights[j] / gauss_weights[i]
        end
    end

    # Calculate Vandermondes
    lobatto_nodes, lobatto_weights = gauss_lobatto_nodes_weights(n_nodes)
    gauss2lobatto = polynomial_interpolation_matrix(gauss_nodes, lobatto_nodes)
    lobatto2gauss = polynomial_interpolation_matrix(lobatto_nodes, gauss_nodes)

    return gauss2lobatto * operator * lobatto2gauss
end

# Calculate reverse projection matrix for discrete L2 projection from upper to large (Gauss-Lobatto
# version)
function calc_reverse_upper(n_nodes, ::Val{:gauss_lobatto})
    # Calculate nodes, weights, and barycentric weights
    nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(1 / 2 * (nodes[j] + 1), nodes, wbary)
        for i in 1:n_nodes
            operator[i, j] = 1 / 2 * poly[i] * weights[j] / weights[i]
        end
    end

    return operator
end

# Calculate reverse projection matrix for discrete L2 projection from lower to large (Gauss-Lobatto
# version)
function calc_reverse_lower(n_nodes, ::Val{:gauss_lobatto})
    # Calculate nodes, weights, and barycentric weights
    nodes, weights = gauss_lobatto_nodes_weights(n_nodes)
    wbary = barycentric_weights(nodes)

    # Calculate projection matrix (actually: discrete L2 projection with errors)
    operator = zeros(n_nodes, n_nodes)
    for j in 1:n_nodes
        poly = lagrange_interpolating_polynomials(1 / 2 * (nodes[j] - 1), nodes, wbary)
        for i in 1:n_nodes
            operator[i, j] = 1 / 2 * poly[i] * weights[j] / weights[i]
        end
    end

    return operator
end
end # @muladd
