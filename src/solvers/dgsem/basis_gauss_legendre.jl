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
                          BoundaryMatrix <: AbstractMatrix{RealT},
                          DerivativeMatrix <: AbstractMatrix{RealT}} <:
       AbstractBasisSBP{RealT}
    nodes::VectorT
    weights::VectorT
    inverse_weights::VectorT

    inverse_vandermonde_legendre::InverseVandermondeLegendre
    boundary_interpolation::BoundaryMatrix # lhat

    derivative_matrix::DerivativeMatrix # strong form derivative matrix
    derivative_split::DerivativeMatrix # strong form derivative matrix minus boundary terms
    derivative_split_transpose::DerivativeMatrix # transpose of `derivative_split`
    derivative_dhat::DerivativeMatrix # weak form matrix "dhat",
    # negative adjoint wrt the SBP dot product
end

function GaussLegendreBasis(RealT, polydeg::Integer)
    nnodes_ = polydeg + 1

    # compute everything using `Float64` by default
    nodes_, weights_ = gauss_nodes_weights(nnodes_)
    inverse_weights_ = inv.(weights_)

    _, inverse_vandermonde_legendre_ = vandermonde_legendre(nodes_)

    boundary_interpolation_ = zeros(nnodes_, 2)
    boundary_interpolation_[:, 1] = calc_lhat(-1.0, nodes_, weights_)
    boundary_interpolation_[:, 2] = calc_lhat(1.0, nodes_, weights_)

    derivative_matrix_ = polynomial_derivative_matrix(nodes_)
    derivative_split_ = calc_dsplit(nodes_, weights_)
    derivative_split_transpose_ = Matrix(derivative_split_')
    derivative_dhat_ = calc_dhat(nodes_, weights_)

    # type conversions to get the requested real type and enable possible
    # optimizations of runtime performance and latency
    nodes = SVector{nnodes_, RealT}(nodes_)
    weights = SVector{nnodes_, RealT}(weights_)
    inverse_weights = SVector{nnodes_, RealT}(inverse_weights_)

    inverse_vandermonde_legendre = convert.(RealT, inverse_vandermonde_legendre_)
    boundary_interpolation = convert.(RealT, boundary_interpolation_)

    # Usually as fast as `SMatrix` (when using `let` in the volume integral/`@threaded`)
    derivative_matrix = Matrix{RealT}(derivative_matrix_)
    derivative_split = Matrix{RealT}(derivative_split_)
    derivative_split_transpose = Matrix{RealT}(derivative_split_transpose_)
    derivative_dhat = Matrix{RealT}(derivative_dhat_)

    return GaussLegendreBasis{RealT, nnodes_, typeof(nodes),
                              typeof(inverse_vandermonde_legendre),
                              typeof(boundary_interpolation),
                              typeof(derivative_matrix)}(nodes, weights,
                                                         inverse_weights,
                                                         inverse_vandermonde_legendre,
                                                         boundary_interpolation,
                                                         derivative_matrix,
                                                         derivative_split,
                                                         derivative_split_transpose,
                                                         derivative_dhat)
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

function Base.:(==)(b1::GaussLegendreBasis, b2::GaussLegendreBasis)
    if typeof(b1) != typeof(b2)
        return false
    end

    for field in fieldnames(typeof(b1))
        if getfield(b1, field) != getfield(b2, field)
            return false
        end
    end

    return true
end

@inline Base.real(basis::GaussLegendreBasis{RealT}) where {RealT} = RealT

@inline function nnodes(basis::GaussLegendreBasis{RealT, NNODES}) where {RealT, NNODES}
    NNODES
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

# Return the first/last weight of the quadrature associated with `basis`.
# Since the mass matrix for nodal Gauss-Legendre bases is diagonal,
# these weights are the only coefficients necessary for the scaling of
# surface terms/integrals in DGSEM.
left_boundary_weight(basis::GaussLegendreBasis) = first(basis.weights)
right_boundary_weight(basis::GaussLegendreBasis) = last(basis.weights)

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
end

function Base.show(io::IO, ::MIME"text/plain", analyzer::GaussLegendreAnalyzer)
    @nospecialize analyzer # reduce precompilation time

    print(io, "GaussLegendreAnalyzer{", real(analyzer),
          "} with polynomials of degree ", polydeg(analyzer))
end

@inline Base.real(analyzer::GaussLegendreAnalyzer{RealT}) where {RealT} = RealT

@inline function nnodes(analyzer::GaussLegendreAnalyzer{RealT, NNODES}) where {RealT,
                                                                               NNODES}
    NNODES
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
