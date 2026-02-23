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
    derivative_split::DerivativeMatrix # strong form derivative matrix minus boundary terms
    derivative_hat::DerivativeMatrix # weak form matrix "dhat", negative adjoint wrt the SBP dot product

    # Required for Gauss-Legendre nodes (non-trivial interpolation to the boundaries)
    boundary_interpolation::BoundaryMatrix # L
    boundary_interpolation_inverse_weights::BoundaryMatrix # M^{-1} * L = Lhat
end

function GaussLegendreBasis(RealT, polydeg::Integer)
    nnodes_ = polydeg + 1

    # compute everything using `Float64` by default
    nodes_, weights_ = gauss_nodes_weights(nnodes_)
    inverse_weights_ = inv.(weights_)

    _, inverse_vandermonde_legendre = vandermonde_legendre(nodes_)

    derivative_matrix = polynomial_derivative_matrix(nodes_)
    derivative_split = calc_Dsplit(derivative_matrix, weights_)
    derivative_hat = calc_Dhat(derivative_matrix, weights_)

    # type conversions to get the requested real type and enable possible
    # optimizations of runtime performance and latency
    nodes = SVector{nnodes_, RealT}(nodes_)
    weights = SVector{nnodes_, RealT}(weights_)
    inverse_weights = SVector{nnodes_, RealT}(inverse_weights_)

    boundary_interpolation = zeros(nnodes_, 2)
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
                                                              derivative_split,
                                                              derivative_hat,
                                                              boundary_interpolation,
                                                              boundary_interpolation_inverse_weights)
end

GaussLegendreBasis(polydeg::Integer) = GaussLegendreBasis(Float64, polydeg)

function Base.show(io::IO, basis::GaussLegendreBasis)
    @nospecialize basis # reduce precompilation time

    print(io, "GaussLegendreBasis{", real(basis), "}(polydeg=", polydeg(basis), ")")
end

function Base.show(io::IO, ::MIME"text/plain", basis::GaussLegendreBasis)
    @nospecialize basis # reduce precompilation time

    print(io, "GaussLegendreBasis{", real(basis), "} with polynomials of degree ",
          polydeg(basis))
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

# TODO: Not yet implemented
function MortarL2(basis::GaussLegendreBasis)
    return nothing
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
