
"""
    LobattoLegendreBasis([RealT=Float64,] polydeg::Integer)

Create a nodal Lobatto-Legendre basis for polynomials of degree `polydeg`.
"""
struct LobattoLegendreBasis{RealT<:Real, NNODES,
                            InverseVandermondeLegendre<:AbstractMatrix{RealT},
                            BoundaryMatrix<:AbstractMatrix{RealT},
                            DerivativeMatrix<:AbstractMatrix{RealT}} <: AbstractBasisSBP{RealT}
  nodes          ::SVector{NNODES, RealT}
  weights        ::SVector{NNODES, RealT}
  inverse_weights::SVector{NNODES, RealT}

  inverse_vandermonde_legendre::InverseVandermondeLegendre
  boundary_interpolation      ::BoundaryMatrix # lhat

  derivative_matrix         ::DerivativeMatrix # dsplit
  derivative_split          ::DerivativeMatrix # dsplit
  derivative_split_transpose::DerivativeMatrix # dsplit_transposed
  derivative_dhat           ::DerivativeMatrix # dhat, neg. adjoint wrt the SBP dot product
end

function LobattoLegendreBasis(RealT, polydeg::Integer)
  nnodes_ = polydeg + 1
  nodes, weights = gauss_lobatto_nodes_weights(nnodes_)
  inverse_weights = inv.(weights)

  _, inverse_vandermonde_legendre = vandermonde_legendre(nodes)

  boundary_interpolation = zeros(nnodes_, 2)
  boundary_interpolation[:, 1] = calc_lhat(-1.0, nodes, weights)
  boundary_interpolation[:, 2] = calc_lhat( 1.0, nodes, weights)

  derivative_matrix          = polynomial_derivative_matrix(nodes)
  derivative_split           = calc_dsplit(nodes, weights)
  derivative_split_transpose = Matrix(derivative_split')
  derivative_dhat            = calc_dhat(nodes, weights)

  # type conversions to make use of StaticArrays etc.
  nodes           = SVector{nnodes_}(convert.(RealT, nodes))
  weights         = SVector{nnodes_}(convert.(RealT, weights))
  inverse_weights = SVector{nnodes_}(convert.(RealT, inverse_weights))

  inverse_vandermonde_legendre = convert.(RealT, inverse_vandermonde_legendre)
  boundary_interpolation       = SMatrix{nnodes_, 2}(convert.(RealT, boundary_interpolation))

  derivative_matrix          = SMatrix{nnodes_, nnodes_}(convert.(RealT, derivative_matrix))
  derivative_split           = SMatrix{nnodes_, nnodes_}(convert.(RealT, derivative_split))
  derivative_split_transpose = SMatrix{nnodes_, nnodes_}(convert.(RealT, derivative_split_transpose))
  derivative_dhat            = SMatrix{nnodes_, nnodes_}(convert.(RealT, derivative_dhat))

  return LobattoLegendreBasis{RealT, nnodes_, typeof(inverse_vandermonde_legendre), typeof(boundary_interpolation), typeof(derivative_matrix)}(
    nodes, weights, inverse_weights,
    inverse_vandermonde_legendre, boundary_interpolation,
    derivative_matrix, derivative_split, derivative_split_transpose, derivative_dhat
  )
end

LobattoLegendreBasis(polydeg::Integer) = LobattoLegendreBasis(Float64, polydeg)

function Base.show(io::IO, @nospecialize basis::LobattoLegendreBasis)
  print(io, "LobattoLegendreBasis{", real(basis), "}(polydeg=", polydeg(basis), ")")
end
function Base.show(io::IO, ::MIME"text/plain", @nospecialize basis::LobattoLegendreBasis)
  print(io, "LobattoLegendreBasis{", real(basis), "} with polynomials of degree ", polydeg(basis))
end

@inline Base.real(basis::LobattoLegendreBasis{RealT}) where {RealT} = RealT

@inline nnodes(basis::LobattoLegendreBasis{RealT, NNODES}) where {RealT, NNODES} = NNODES

@inline polydeg(basis::LobattoLegendreBasis) = nnodes(basis) - 1



struct LobattoLegendreMortarL2{RealT<:Real, NNODES, MortarMatrix<:AbstractMatrix{RealT}} <: AbstractMortarL2{RealT}
  forward_upper::MortarMatrix
  forward_lower::MortarMatrix
  reverse_upper::MortarMatrix
  reverse_lower::MortarMatrix
end

function MortarL2(basis::LobattoLegendreBasis)
  RealT = real(basis)
  NNODES = nnodes(basis)

  forward_upper = calc_forward_upper(NNODES)
  forward_lower = calc_forward_lower(NNODES)
  reverse_upper = calc_reverse_upper(NNODES, Val(:gauss))
  reverse_lower = calc_reverse_lower(NNODES, Val(:gauss))

  # type conversions to make use of StaticArrays etc.
  forward_upper = SMatrix{NNODES, NNODES}(convert.(RealT, forward_upper))
  forward_lower = SMatrix{NNODES, NNODES}(convert.(RealT, forward_lower))
  reverse_upper = SMatrix{NNODES, NNODES}(convert.(RealT, reverse_upper))
  reverse_lower = SMatrix{NNODES, NNODES}(convert.(RealT, reverse_lower))

  LobattoLegendreMortarL2{RealT, NNODES, typeof(forward_upper)}(
    forward_upper, forward_lower,
    reverse_upper, reverse_lower)
end

function Base.show(io::IO, @nospecialize mortar::LobattoLegendreMortarL2)
  print(io, "LobattoLegendreMortarL2{", real(mortar), "}(polydeg=", polydeg(mortar), ")")
end
function Base.show(io::IO, ::MIME"text/plain", @nospecialize mortar::LobattoLegendreMortarL2)
  print(io, "LobattoLegendreMortarL2{", real(mortar), "} with polynomials of degree ", polydeg(mortar))
end

@inline Base.real(mortar::LobattoLegendreMortarL2{RealT}) where {RealT} = RealT

@inline nnodes(mortar::LobattoLegendreMortarL2{RealT, NNODES}) where {RealT, NNODES} = NNODES

@inline polydeg(mortar::LobattoLegendreMortarL2) = nnodes(mortar) - 1


# TODO: We can create EC mortars along the lines of the following implementation.
# abstract type AbstractMortarEC{RealT} <: AbstractMortar{RealT} end

# struct LobattoLegendreMortarEC{RealT<:Real, NNODES, MortarMatrix<:AbstractMatrix{RealT}, SurfaceFlux} <: AbstractMortarEC{RealT}
#   forward_upper::MortarMatrix
#   forward_lower::MortarMatrix
#   reverse_upper::MortarMatrix
#   reverse_lower::MortarMatrix
#   surface_flux::SurfaceFlux
# end

# function MortarEC(basis::LobattoLegendreBasis{RealT}, surface_flux)
#   forward_upper   = calc_forward_upper(n_nodes)
#   forward_lower   = calc_forward_lower(n_nodes)
#   l2reverse_upper = calc_reverse_upper(n_nodes, Val(:gauss_lobatto))
#   l2reverse_lower = calc_reverse_lower(n_nodes, Val(:gauss_lobatto))

#   # type conversions to make use of StaticArrays etc.
#   nnodes_ = nnodes(basis)
#   forward_upper   = SMatrix{nnodes_, nnodes_}(forward_upper)
#   forward_lower   = SMatrix{nnodes_, nnodes_}(forward_lower)
#   l2reverse_upper = SMatrix{nnodes_, nnodes_}(l2reverse_upper)
#   l2reverse_lower = SMatrix{nnodes_, nnodes_}(l2reverse_lower)

#   LobattoLegendreMortarEC{RealT, nnodes_, typeof(forward_upper), typeof(surface_flux)}(
#     forward_upper, forward_lower,
#     l2reverse_upper, l2reverse_lower,
#     surface_flux
#   )
# end

# @inline nnodes(mortar::LobattoLegendreMortarEC{RealT, NNODES}) = NNODES



struct LobattoLegendreAnalyzer{RealT<:Real, NNODES, Vandermonde<:AbstractMatrix{RealT}} <: SolutionAnalyzer{RealT}
  nodes  ::SVector{NNODES, RealT}
  weights::SVector{NNODES, RealT}
  vandermonde::Vandermonde
end

function SolutionAnalyzer(basis::LobattoLegendreBasis{RealT}; analysis_polydeg=2*polydeg(basis)) where {RealT}
  nnodes_ = analysis_polydeg + 1
  nodes, weights = gauss_lobatto_nodes_weights(nnodes_)

  vandermonde = polynomial_interpolation_matrix(basis.nodes, nodes)

  # type conversions to make use of StaticArrays etc.
  nodes   = SVector{nnodes_}(convert.(RealT, nodes))
  weights = SVector{nnodes_}(convert.(RealT, weights))

  vandermonde = convert.(RealT, vandermonde)

  return LobattoLegendreAnalyzer{RealT, nnodes_, typeof(vandermonde)}(
    nodes, weights, vandermonde)
end

function Base.show(io::IO, @nospecialize analyzer::LobattoLegendreAnalyzer)
  print(io, "LobattoLegendreAnalyzer{", real(analyzer), "}(polydeg=", polydeg(analyzer), ")")
end
function Base.show(io::IO, ::MIME"text/plain", @nospecialize analyzer::LobattoLegendreAnalyzer)
  print(io, "LobattoLegendreAnalyzer{", real(analyzer), "} with polynomials of degree ", polydeg(analyzer))
end

@inline Base.real(analyzer::LobattoLegendreAnalyzer{RealT}) where {RealT} = RealT

@inline nnodes(analyzer::LobattoLegendreAnalyzer{RealT, NNODES}) where {RealT, NNODES} = NNODES
@inline eachnode(analyzer::LobattoLegendreAnalyzer) = Base.OneTo(nnodes(analyzer))

@inline polydeg(analyzer::LobattoLegendreAnalyzer) = nnodes(analyzer) - 1



struct LobattoLegendreAdaptorL2{RealT<:Real, NNODES, MortarMatrix<:AbstractMatrix{RealT}} <: AdaptorL2{RealT}
  forward_upper::MortarMatrix
  forward_lower::MortarMatrix
  reverse_upper::MortarMatrix
  reverse_lower::MortarMatrix
end

function AdaptorL2(basis::LobattoLegendreBasis{RealT}) where {RealT}
  nnodes_ = nnodes(basis)
  forward_upper   = calc_forward_upper(nnodes_)
  forward_lower   = calc_forward_lower(nnodes_)
  l2reverse_upper = calc_reverse_upper(nnodes_, Val(:gauss))
  l2reverse_lower = calc_reverse_lower(nnodes_, Val(:gauss))

  # type conversions to make use of StaticArrays etc.
  forward_upper   = SMatrix{nnodes_, nnodes_}(convert.(RealT, forward_upper))
  forward_lower   = SMatrix{nnodes_, nnodes_}(convert.(RealT, forward_lower))
  l2reverse_upper = SMatrix{nnodes_, nnodes_}(convert.(RealT, l2reverse_upper))
  l2reverse_lower = SMatrix{nnodes_, nnodes_}(convert.(RealT, l2reverse_lower))

  LobattoLegendreAdaptorL2{RealT, nnodes_, typeof(forward_upper)}(
    forward_upper, forward_lower,
    l2reverse_upper, l2reverse_lower)
end

function Base.show(io::IO, @nospecialize adaptor::LobattoLegendreAdaptorL2)
  print(io, "LobattoLegendreAdaptorL2{", real(adaptor), "}(polydeg=", polydeg(adaptor), ")")
end
function Base.show(io::IO, ::MIME"text/plain", @nospecialize adaptor::LobattoLegendreAdaptorL2)
  print(io, "LobattoLegendreAdaptorL2{", real(adaptor), "} with polynomials of degree ", polydeg(adaptor))
end

@inline Base.real(adaptor::LobattoLegendreAdaptorL2{RealT}) where {RealT} = RealT

@inline nnodes(adaptor::LobattoLegendreAdaptorL2{RealT, NNODES}) where {RealT, NNODES} = NNODES

@inline polydeg(adaptor::LobattoLegendreAdaptorL2) = nnodes(adaptor) - 1


###############################################################################
# Polynomial derivative and interpolation functions

# TODO: Taal refactor, allow other RealT below and adapt constructors above accordingly

# Calculate the Dhat matrix
function calc_dhat(nodes, weights)
  n_nodes = length(nodes)
  dhat = Matrix(polynomial_derivative_matrix(nodes)')

  for n in 1:n_nodes, j in 1:n_nodes
    dhat[j, n] *= -weights[n] / weights[j]
  end

  return dhat
end


# Calculate the Dsplit matrix for split-form differentiation: dplit = 2D - M⁻¹B
function calc_dsplit(nodes, weights)
  # Start with 2 x the normal D matrix
  dsplit = 2 .* polynomial_derivative_matrix(nodes)

  # Modify to account for
  dsplit[  1,   1] += 1 / weights[1]
  dsplit[end, end] -= 1 / weights[end]

  return dsplit
end


# Calculate the polynomial derivative matrix D
function polynomial_derivative_matrix(nodes)
  n_nodes = length(nodes)
  d = zeros(n_nodes, n_nodes)
  wbary = barycentric_weights(nodes)

  for i in 1:n_nodes, j in 1:n_nodes
    if j != i
      d[i, j] = wbary[j] / wbary[i] * 1 / (nodes[i] - nodes[j])
      d[i, i] -= d[i, j]
    end
  end

  return d
end


# Calculate and interpolation matrix (Vandermonde matrix) between two given sets of nodes
function polynomial_interpolation_matrix(nodes_in, nodes_out)
  n_nodes_in = length(nodes_in)
  n_nodes_out = length(nodes_out)
  wbary_in = barycentric_weights(nodes_in)
  vdm = zeros(n_nodes_out, n_nodes_in)

  for k in 1:n_nodes_out
    match = false
    for j in 1:n_nodes_in
      if isapprox(nodes_out[k], nodes_in[j], rtol=eps())
        match = true
        vdm[k, j] = 1
      end
    end

    if match == false
      s = 0.0
      for j in 1:n_nodes_in
        t = wbary_in[j] / (nodes_out[k] - nodes_in[j])
        vdm[k, j] = t
        s += t
      end
      for j in 1:n_nodes_in
        vdm[k, j] = vdm[k, j] / s
      end
    end
  end

  return vdm
end


# Calculate the barycentric weights for a given node distribution.
function barycentric_weights(nodes)
  n_nodes = length(nodes)
  weights = ones(n_nodes)

  for j = 2:n_nodes, k = 1:(j-1)
    weights[k] *= nodes[k] - nodes[j]
    weights[j] *= nodes[j] - nodes[k]
  end

  for j in 1:n_nodes
    weights[j] = 1 / weights[j]
  end

  return weights
end


# Calculate Lhat.
function calc_lhat(x, nodes, weights)
  n_nodes = length(nodes)
  wbary = barycentric_weights(nodes)

  lhat = lagrange_interpolating_polynomials(x, nodes, wbary)

  for i in 1:n_nodes
    lhat[i] /= weights[i]
  end

  return lhat
end


# Calculate Lagrange polynomials for a given node distribution.
function lagrange_interpolating_polynomials(x, nodes, wbary)
  n_nodes = length(nodes)
  polynomials = zeros(n_nodes)

  for i in 1:n_nodes
    if isapprox(x, nodes[i], rtol=eps(x))
      polynomials[i] = 1
      return polynomials
    end
  end

  for i in 1:n_nodes
    polynomials[i] = wbary[i] / (x - nodes[i])
  end
  total = sum(polynomials)

  for i in 1:n_nodes
    polynomials[i] /= total
  end

  return polynomials
end


# From FLUXO (but really from blue book by Kopriva)
function gauss_lobatto_nodes_weights(n_nodes::Integer)
  # From Kopriva's book
  n_iterations = 10
  tolerance = 1e-15

  # Initialize output
  nodes = zeros(n_nodes)
  weights = zeros(n_nodes)

  # Get polynomial degree for convenience
  N = n_nodes - 1

  # Calculate values at boundary
  nodes[1] = -1.0
  nodes[end] = 1.0
  weights[1] = 2 / (N * (N + 1))
  weights[end] = weights[1]

  # Calculate interior values
  if N > 1
    cont1 = pi/N
    cont2 = 3/(8 * N * pi)

    # Use symmetry -> only left side is computed
    for i in 1:(div(N + 1, 2) - 1)
      # Calculate node
      # Initial guess for Newton method
      nodes[i+1] = -cos(cont1*(i+0.25) - cont2/(i+0.25))

      # Newton iteration to find root of Legendre polynomial (= integration node)
      for k in 0:n_iterations
        q, qder, _ = calc_q_and_l(N, nodes[i+1])
        dx = -q/qder
        nodes[i+1] += dx
        if abs(dx) < tolerance * abs(nodes[i+1])
          break
        end
      end

      # Calculate weight
      _, _, L = calc_q_and_l(N, nodes[i+1])
      weights[i+1] = weights[1] / L^2

      # Set nodes and weights according to symmetry properties
      nodes[N+1-i] = -nodes[i+1]
      weights[N+1-i] = weights[i+1]
    end
  end

  # If odd number of nodes, set center node to origin (= 0.0) and calculate weight
  if n_nodes % 2 == 1
    _, _, L = calc_q_and_l(N, 0)
    nodes[div(N, 2) + 1] = 0.0
    weights[div(N, 2) + 1] = weights[1] / L^2
  end

  return nodes, weights
end


# From FLUXO (but really from blue book by Kopriva)
function calc_q_and_l(N::Integer, x::Float64)
  L_Nm2 = 1.0
  L_Nm1 = x
  Lder_Nm2 = 0.0
  Lder_Nm1 = 1.0

  local L
  for i in 2:N
    L = ((2 * i - 1) * x * L_Nm1 - (i - 1) * L_Nm2) / i
    Lder = Lder_Nm2 + (2 * i - 1) * L_Nm1
    L_Nm2 = L_Nm1
    L_Nm1 = L
    Lder_Nm2 = Lder_Nm1
    Lder_Nm1 = Lder
  end

  q = (2 * N + 1)/(N + 1) * (x * L - L_Nm2)
  qder = (2 * N + 1) * L

  return q, qder, L
end
calc_q_and_l(N::Integer, x::Real) = calc_q_and_l(N, convert(Float64, x))


# From FLUXO (but really from blue book by Kopriva)
function gauss_nodes_weights(n_nodes::Integer)
  # From Kopriva's book
  n_iterations = 10
  tolerance = 1e-15

  # Initialize output
  nodes = ones(n_nodes) * 1000
  weights = zeros(n_nodes)

  # Get polynomial degree for convenience
  N = n_nodes - 1
  if N == 0
    nodes .= 0.0
    weights .= 2.0
    return nodes, weights
  elseif N == 1
    nodes[1] = -sqrt(1/3)
    nodes[end] = -nodes[1]
    weights .= 1.0
    return nodes, weights
  else # N > 1
    # Use symmetry property of the roots of the Legendre polynomials
    for i in 0:(div(N + 1, 2) - 1)
      # Starting guess for Newton method
      nodes[i+1] = -cos(pi / (2 * N + 2) * (2 * i + 1))

      # Newton iteration to find root of Legendre polynomial (= integration node)
      for k in 0:n_iterations
        poly, deriv = legendre_polynomial_and_derivative(N + 1, nodes[i+1])
        dx = -poly / deriv
        nodes[i+1] += dx
        if abs(dx) < tolerance * abs(nodes[i+1])
          break
        end
      end

      # Calculate weight
      poly, deriv = legendre_polynomial_and_derivative(N + 1, nodes[i+1])
      weights[i+1] = (2 * N + 3) / ((1 - nodes[i+1]^2) * deriv^2)

      # Set nodes and weights according to symmetry properties
      nodes[N+1-i] = -nodes[i+1]
      weights[N+1-i] = weights[i+1]
    end

    # If odd number of nodes, set center node to origin (= 0.0) and calculate weight
    if n_nodes % 2 == 1
      poly, deriv = legendre_polynomial_and_derivative(N + 1, 0.0)
      nodes[div(N, 2) + 1] = 0.0
      weights[div(N, 2) + 1] = (2 * N + 3) / deriv^2
    end

    return nodes, weights
  end
end


# From FLUXO (but really from blue book by Kopriva)
function legendre_polynomial_and_derivative(N::Int, x::Real)
  if N == 0
    poly = 1.0
    deriv = 0.0
  elseif N == 1
    poly = convert(Float64, x)
    deriv = 1.0
  else
    poly_Nm2 = 1.0
    poly_Nm1 = convert(Float64, x)
    deriv_Nm2 = 0.0
    deriv_Nm1 = 1.0

    poly = 0.0
    deriv = 0.0
    for i in 2:N
      poly = ((2*i-1) * x * poly_Nm1 - (i-1) * poly_Nm2) / i
      deriv=deriv_Nm2 + (2*i-1)*poly_Nm1
      poly_Nm2=poly_Nm1
      poly_Nm1=poly
      deriv_Nm2=deriv_Nm1
      deriv_Nm1=deriv
    end
  end

  # Normalize
  poly = poly * sqrt(N+0.5)
  deriv = deriv * sqrt(N+0.5)

  return poly, deriv
end


# Calculate Legendre vandermonde matrix and its inverse
function vandermonde_legendre(nodes, N)
  n_nodes = length(nodes)
  n_modes = N + 1
  vandermonde = zeros(n_nodes, n_modes)

  for i in 1:n_nodes
    for m in 1:n_modes
      vandermonde[i, m], _ = legendre_polynomial_and_derivative(m-1, nodes[i])
    end
  end
  # for very high polynomial degree, this is not well conditioned
  inverse_vandermonde = inv(vandermonde)
  return vandermonde, inverse_vandermonde
end
vandermonde_legendre(nodes) = vandermonde_legendre(nodes, length(nodes) - 1)
