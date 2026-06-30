# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    GaussLegendreBasis([RealT=Float64,] polydeg::Integer)

Create a nodal Gauss-Legendre basis for polynomials of degree `polydeg`.
"""
struct GaussLegendreBasis{RealT <: Real, NNODES,
                          VectorT <: AbstractVector{RealT},
                          InverseVandermondeLegendre <: AbstractMatrix{RealT},
                          DerivativeMatrix <: AbstractMatrix{RealT},
                          BoundaryMatrix <: AbstractMatrix{RealT}} <:
       AbstractBasisSBP{RealT}
    nodes::VectorT
    weights::VectorT
    inverse_weights::VectorT

    inverse_vandermonde_legendre::InverseVandermondeLegendre

    derivative_matrix::DerivativeMatrix # strong form derivative matrix
    # `derivative_split` currently not implemented since
    # Flux-Differencing is not supported for Gauss-Legendre DGSEM at the moment.
    derivative_hat::DerivativeMatrix # weak form matrix "dhat", negative adjoint wrt the SBP dot product

    # Required for Gauss-Legendre nodes (non-trivial interpolation to the boundaries)
    boundary_interpolation::BoundaryMatrix # L
    boundary_interpolation_inverse_weights::BoundaryMatrix # M^{-1} * L = Lhat
end

function GaussLegendreBasis(RealT, polydeg::Integer)
    nnodes_ = polydeg + 1

    nodes_, weights_ = gauss_nodes_weights(nnodes_, RealT)
    inverse_weights_ = inv.(weights_)

    _, inverse_vandermonde_legendre = vandermonde_legendre(nodes_, RealT)

    derivative_matrix = polynomial_derivative_matrix(nodes_)
    derivative_hat = calc_Dhat(derivative_matrix, weights_)

    # Type conversions to enable possible optimizations of runtime performance
    # and latency
    nodes = SVector{nnodes_, RealT}(nodes_)
    weights = SVector{nnodes_, RealT}(weights_)
    inverse_weights = SVector{nnodes_, RealT}(inverse_weights_)

    boundary_interpolation = zeros(RealT, nnodes_, 2)
    boundary_interpolation[:, 1] = calc_L(-one(RealT), nodes_, weights_)
    boundary_interpolation[:, 2] = calc_L(one(RealT), nodes_, weights_)

    boundary_interpolation_inverse_weights = copy(boundary_interpolation)
    boundary_interpolation_inverse_weights[:, 1] = calc_Lhat(boundary_interpolation[:,
                                                                                    1],
                                                             weights_)
    boundary_interpolation_inverse_weights[:, 2] = calc_Lhat(boundary_interpolation[:,
                                                                                    2],
                                                             weights_)

    # We keep the matrices above stored using the standard `Matrix` type
    # since this is usually as fast as `SMatrix`
    # (when using `let` in the volume integral/`@threaded`)
    # and reduces latency

    return GaussLegendreBasis{RealT, nnodes_, typeof(nodes),
                              typeof(inverse_vandermonde_legendre),
                              typeof(derivative_matrix),
                              typeof(boundary_interpolation)}(nodes, weights,
                                                              inverse_weights,
                                                              inverse_vandermonde_legendre,
                                                              derivative_matrix,
                                                              derivative_hat,
                                                              boundary_interpolation,
                                                              boundary_interpolation_inverse_weights)
end

GaussLegendreBasis(polydeg::Integer) = GaussLegendreBasis(Float64, polydeg)

function Base.show(io::IO, basis::GaussLegendreBasis)
    @nospecialize basis # reduce precompilation time

    print(io, "GaussLegendreBasis{", real(basis), "}(polydeg=", polydeg(basis), ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", basis::GaussLegendreBasis)
    @nospecialize basis # reduce precompilation time

    print(io, "GaussLegendreBasis{", real(basis), "} with polynomials of degree ",
          polydeg(basis))
    return nothing
end

@inline Base.real(basis::GaussLegendreBasis{RealT}) where {RealT} = RealT

@inline function nnodes(basis::GaussLegendreBasis{RealT, NNODES}) where {RealT, NNODES}
    return NNODES
end

"""
    eachnode(basis::GaussLegendreBasis)

Return an iterator over the indices that specify the location in relevant data structures
for the nodes in `basis`. 
In particular, not the nodes themselves are returned.
"""
@inline eachnode(basis::GaussLegendreBasis) = Base.OneTo(nnodes(basis))

@inline polydeg(basis::GaussLegendreBasis) = nnodes(basis) - 1

@inline get_nodes(basis::GaussLegendreBasis) = basis.nodes

"""
    integrate(f, u, basis::GaussLegendreBasis)

Map the function `f` to the coefficients `u` and integrate with respect to the
quadrature rule given by `basis`.
"""
function integrate(f, u, basis::GaussLegendreBasis)
    @unpack weights = basis

    res = zero(f(first(u)))
    for i in eachindex(u, weights)
        res += f(u[i]) * weights[i]
    end
    return res
end

# TODO: Not yet implemented
function MortarL2(basis::GaussLegendreBasis)
    return nothing
end

"""
    gauss_nodes_weights(n_nodes::Integer, RealT = Float64)

Computes nodes ``x_j`` and weights ``w_j`` for the Gauss-Legendre quadrature.
This implements algorithm 23 "LegendreGaussNodesAndWeights" from the book

- David A. Kopriva, (2009).
  Implementing spectral methods for partial differential equations:
  Algorithms for scientists and engineers.
  [DOI:10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
"""
function gauss_nodes_weights(n_nodes::Integer, RealT = Float64)
    n_iterations = 20
    tolerance = 2 * eps(RealT) # Relative tolerance for Newton iteration

    # Initialize output
    nodes = ones(RealT, n_nodes)
    weights = zeros(RealT, n_nodes)

    # Get polynomial degree for convenience
    N = n_nodes - 1
    if N == 0
        nodes .= 0
        weights .= 2
        return nodes, weights
    elseif N == 1
        nodes[1] = -sqrt(one(RealT) / 3)
        nodes[end] = -nodes[1]
        weights .= 1
        return nodes, weights
    else # N > 1
        # Use symmetry property of the roots of the Legendre polynomials
        for i in 0:(div(N + 1, 2) - 1)
            # Starting guess for Newton method
            nodes[i + 1] = -cospi(one(RealT) / (2 * N + 2) * (2 * i + 1))

            # Newton iteration to find root of Legendre polynomial (= integration node)
            for k in 0:n_iterations
                poly, deriv = legendre_polynomial_and_derivative(N + 1, nodes[i + 1])
                dx = -poly / deriv
                nodes[i + 1] += dx
                if abs(dx) < tolerance * abs(nodes[i + 1])
                    break
                end

                if k == n_iterations
                    @warn "`gauss_nodes_weights` Newton iteration did not converge"
                end
            end

            # Calculate weight
            poly, deriv = legendre_polynomial_and_derivative(N + 1, nodes[i + 1])
            weights[i + 1] = (2 * N + 3) / ((1 - nodes[i + 1]^2) * deriv^2)

            # Set nodes and weights according to symmetry properties
            nodes[N + 1 - i] = -nodes[i + 1]
            weights[N + 1 - i] = weights[i + 1]
        end

        # If odd number of nodes, set center node to origin (= 0.0) and calculate weight
        if n_nodes % 2 == 1
            poly, deriv = legendre_polynomial_and_derivative(N + 1, zero(RealT))
            nodes[div(N, 2) + 1] = 0
            weights[div(N, 2) + 1] = (2 * N + 3) / deriv^2
        end

        return nodes, weights
    end
end

# L(x), where L(x) is the Lagrange polynomial vector at point x.
function calc_L(x, nodes, weights)
    n_nodes = length(nodes)
    wbary = barycentric_weights(nodes)

    return lagrange_interpolating_polynomials(x, nodes, wbary)
end

# Calculate M^{-1} * L(x), where L(x) is the Lagrange polynomial
# vector at point x.
# Not required for the DGSEM with LGL basis, as boundary evaluations
# collapse to boundary node evaluations.
function calc_Lhat(L, weights)
    Lhat = copy(L)
    for i in 1:length(weights)
        Lhat[i] /= weights[i]
    end

    return Lhat
end

struct GaussLegendreAnalyzer{RealT <: Real, NNODES,
                             VectorT <: AbstractVector{RealT},
                             Vandermonde <: AbstractMatrix{RealT}} <:
       SolutionAnalyzer{RealT}
    nodes::VectorT
    weights::VectorT
    vandermonde::Vandermonde
end

function SolutionAnalyzer(basis::GaussLegendreBasis;
                          analysis_polydeg = 2 * polydeg(basis))
    RealT = real(basis)
    nnodes_ = analysis_polydeg + 1

    # compute everything using `Float64` by default
    nodes_, weights_ = gauss_nodes_weights(nnodes_)
    vandermonde_ = polynomial_interpolation_matrix(get_nodes(basis), nodes_)

    # type conversions to get the requested real type and enable possible
    # optimizations of runtime performance and latency
    nodes = SVector{nnodes_, RealT}(nodes_)
    weights = SVector{nnodes_, RealT}(weights_)

    vandermonde = Matrix{RealT}(vandermonde_)

    return GaussLegendreAnalyzer{RealT, nnodes_, typeof(nodes), typeof(vandermonde)}(nodes,
                                                                                     weights,
                                                                                     vandermonde)
end

function Base.show(io::IO, analyzer::GaussLegendreAnalyzer)
    @nospecialize analyzer # reduce precompilation time

    print(io, "GaussLegendreAnalyzer{", real(analyzer), "}(polydeg=",
          polydeg(analyzer), ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", analyzer::GaussLegendreAnalyzer)
    @nospecialize analyzer # reduce precompilation time

    print(io, "GaussLegendreAnalyzer{", real(analyzer),
          "} with polynomials of degree ", polydeg(analyzer))
    return nothing
end

@inline Base.real(analyzer::GaussLegendreAnalyzer{RealT}) where {RealT} = RealT

@inline function nnodes(analyzer::GaussLegendreAnalyzer{RealT, NNODES}) where {RealT,
                                                                               NNODES}
    return NNODES
end

"""
    eachnode(analyzer::GaussLegendreAnalyzer)

Return an iterator over the indices that specify the location in relevant data structures
for the nodes in `analyzer`. 
In particular, not the nodes themselves are returned.
"""
@inline eachnode(analyzer::GaussLegendreAnalyzer) = Base.OneTo(nnodes(analyzer))

@inline polydeg(analyzer::GaussLegendreAnalyzer) = nnodes(analyzer) - 1
end # @muladd
