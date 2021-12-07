
"""
    DGMulti(approximation_type::AbstractDerivativeOperator;
            element_type::AbstractElemShape,
            surface_flux=flux_central,
            surface_integral=SurfaceIntegralWeakForm(surface_flux),
            volume_integral=VolumeIntegralWeakForm(),
            kwargs...)

Create a summation by parts (SBP) discretization on the given `element_type`
using a tensor product structure.

For more info, see the documentations of
[StartUpDG.jl](https://jlchan.github.io/StartUpDG.jl/dev/)
and
[SummationByPartsOperators.jl](https://ranocha.de/SummationByPartsOperators.jl/stable/).
"""
function DGMulti(approximation_type::AbstractDerivativeOperator;
                 element_type::AbstractElemShape,
                 surface_flux=flux_central,
                 surface_integral=SurfaceIntegralWeakForm(surface_flux),
                 volume_integral=VolumeIntegralWeakForm(),
                 kwargs...)

  rd = RefElemData(element_type, approximation_type; kwargs...)
  return DG(rd, nothing #= mortar =#, surface_integral, volume_integral)
end

function DGMulti(element_type::AbstractElemShape,
                 approximation_type::AbstractDerivativeOperator,
                 volume_integral,
                 surface_integral;
                 kwargs...)

  DGMulti(approximation_type, element_type=element_type,
          surface_integral=surface_integral, volume_integral=volume_integral)
end


function StartUpDG.RefElemData(element_type::Line,
                               D::AbstractDerivativeOperator;
                               tol = 100*eps())

  approximation_type = D
  N = SummationByPartsOperators.accuracy_order(D) # kind of polynomial degree

  # StartUpDG assumes nodes from -1 to +1
  nodes_1d = collect(grid(D))
  weights_1d = diag(SummationByPartsOperators.mass_matrix(D))
  xmin, xmax = extrema(nodes_1d)
  factor = 2 / (xmax - xmin)
  @. nodes_1d = factor * (nodes_1d - xmin) - 1
  @. weights_1d = factor * weights_1d

  # volume
  rq = r = nodes_1d
  wq = weights_1d
  Dr = droptol!(inv(factor) * sparse(D), tol)
  M = Diagonal(wq)
  Pq = LinearAlgebra.I
  Vq = LinearAlgebra.I

  VDM = nothing # unused generalized Vandermonde matrix

  rst = (r,)
  rstq = (rq,)
  Drst = (Dr,)

  # face
  face_vertices = StartUpDG.face_vertices(element_type)
  face_mask = [1, length(nodes_1d)]

  rf = [-1.0; 1.0]
  nrJ = [-1.0; 1.0]
  wf = [1.0; 1.0]
  Vf = sparse([1, 2], [1, length(nodes_1d)], [1.0, 1.0])
  LIFT = Diagonal(wq) \ (Vf' * Diagonal(wf))

  rstf = (rf,)
  nrstJ = (nrJ,)

  # low order interpolation nodes
  r1 = StartUpDG.nodes(element_type, 1)
  V1 = StartUpDG.vandermonde(element_type, 1, r) / StartUpDG.vandermonde(element_type, 1, r1)

  return RefElemData(
    element_type, approximation_type, N,
    face_vertices, V1,
    rst, VDM, face_mask,
    N, rst, LinearAlgebra.I, # plotting
    rstq, wq, Vq, # quadrature
    rstf, wf, Vf, nrstJ, # faces
    M, Pq, Drst, LIFT)
end


function StartUpDG.RefElemData(element_type::Quad,
                               D::AbstractDerivativeOperator;
                               tol = 100*eps())

  approximation_type = D
  N = SummationByPartsOperators.accuracy_order(D) # kind of polynomial degree

  # StartUpDG assumes nodes from -1 to +1
  nodes_1d = collect(grid(D))
  weights_1d = diag(SummationByPartsOperators.mass_matrix(D))
  xmin, xmax = extrema(nodes_1d)
  factor = 2 / (xmax - xmin)
  @. nodes_1d = factor * (nodes_1d - xmin) - 1
  @. weights_1d = factor * weights_1d

  # volume
  s, r = vec.(StartUpDG.NodesAndModes.meshgrid(nodes_1d)) # this is to match
                                                          # ordering of nrstJ
  rq = r; sq = s
  wr, ws = vec.(StartUpDG.NodesAndModes.meshgrid(weights_1d))
  wq = wr .* ws
  D_1d = droptol!(inv(factor) * sparse(D), tol)
  I_1d = Diagonal(ones(Bool, length(nodes_1d)))
  Dr = kron(I_1d, D_1d)
  Ds = kron(D_1d, I_1d)
  M = Diagonal(wq)
  Pq = LinearAlgebra.I
  Vq = LinearAlgebra.I

  VDM = nothing # unused generalized Vandermonde matrix

  rst = (r, s)
  rstq = (rq, sq)
  Drst = (Dr, Ds)

  # face
  face_vertices = StartUpDG.face_vertices(element_type)
  face_mask = vcat(StartUpDG.find_face_nodes(element_type, r, s)...)

  rf, sf, wf, nrJ, nsJ = StartUpDG.init_face_data(element_type, N,
    quad_rule_face=(nodes_1d, weights_1d))
  Vf = sparse(eachindex(face_mask), face_mask, ones(Bool, length(face_mask)))
  LIFT = Diagonal(wq) \ (Vf' * Diagonal(wf))

  rstf = (rf, sf)
  nrstJ = (nrJ, nsJ)

  # low order interpolation nodes
  r1, s1 = StartUpDG.nodes(element_type, 1)
  V1 = StartUpDG.vandermonde(element_type, 1, r, s) / StartUpDG.vandermonde(element_type, 1, r1, s1)

  return RefElemData(
    element_type, approximation_type, N,
    face_vertices, V1,
    rst, VDM, face_mask,
    N, rst, LinearAlgebra.I, # plotting
    rstq, wq, Vq, # quadrature
    rstf, wf, Vf, nrstJ, # faces
    M, Pq, Drst, LIFT)
end


function StartUpDG.RefElemData(element_type::Hex,
                               D::AbstractDerivativeOperator;
                               tol = 100*eps())

  approximation_type = D
  N = SummationByPartsOperators.accuracy_order(D) # kind of polynomial degree

  # StartUpDG assumes nodes from -1 to +1
  nodes_1d = collect(grid(D))
  weights_1d = diag(SummationByPartsOperators.mass_matrix(D))
  xmin, xmax = extrema(nodes_1d)
  factor = 2 / (xmax - xmin)
  @. nodes_1d = factor * (nodes_1d - xmin) - 1
  @. weights_1d = factor * weights_1d

  # volume
  # to match ordering of nrstJ
  s, r, t = vec.(StartUpDG.NodesAndModes.meshgrid(nodes_1d, nodes_1d, nodes_1d))
  rq = r; sq = s; tq = t
  wr, ws, wt = vec.(StartUpDG.NodesAndModes.meshgrid(weights_1d, weights_1d, weights_1d))
  wq = wr .* ws .* wt
  D_1d = droptol!(inv(factor) * sparse(D), tol)
  I_1d = Diagonal(ones(Bool, length(nodes_1d)))
  Dr = kron(I_1d, I_1d, D_1d)
  Ds = kron(I_1d, D_1d, I_1d)
  Dt = kron(D_1d, I_1d, I_1d)
  M = Diagonal(wq)
  Pq = LinearAlgebra.I
  Vq = LinearAlgebra.I

  VDM = nothing # unused generalized Vandermonde matrix

  rst = (r, s, t)
  rstq = (rq, sq, tq)
  Drst = (Dr, Ds, Dt)

  # face
  face_vertices = StartUpDG.face_vertices(element_type)
  face_mask = vcat(StartUpDG.find_face_nodes(element_type, r, s, t)...)

  rf, sf, tf, wf, nrJ, nsJ, ntJ = let
    rf, sf = vec.(StartUpDG.NodesAndModes.meshgrid(nodes_1d, nodes_1d))
    wr, ws = vec.(StartUpDG.NodesAndModes.meshgrid(weights_1d, weights_1d))
    wf = wr .* ws
    StartUpDG.init_face_data(element_type, N, quad_rule_face=(rf, sf, wf))
  end
  Vf = sparse(eachindex(face_mask), face_mask, ones(Bool, length(face_mask)))
  LIFT = Diagonal(wq) \ (Vf' * Diagonal(wf))

  rstf = (rf, sf, tf)
  nrstJ = (nrJ, nsJ, ntJ)

  # low order interpolation nodes
  r1, s1, t1 = StartUpDG.nodes(element_type, 1)
  V1 = StartUpDG.vandermonde(element_type, 1, r, s, t) / StartUpDG.vandermonde(element_type, 1, r1, s1, t1)

  return RefElemData(
    element_type, approximation_type, N,
    face_vertices, V1,
    rst, VDM, face_mask,
    N, rst, LinearAlgebra.I, # plotting
    rstq, wq, Vq, # quadrature
    rstf, wf, Vf, nrstJ, # faces
    M, Pq, Drst, LIFT)
end


function Base.show(io::IO, mime::MIME"text/plain", rd::RefElemData{NDIMS, ElementType, ApproximationType}) where {NDIMS, ElementType<:StartUpDG.AbstractElemShape, ApproximationType<:AbstractDerivativeOperator}
  @nospecialize rd
  print(io, "RefElemData for an approximation using an ")
  show(IOContext(io, :compact => true), rd.approximationType)
  print(io, " on $(rd.elementType) element")
end

function Base.show(io::IO, rd::RefElemData{NDIMS, ElementType, ApproximationType}) where {NDIMS, ElementType<:StartUpDG.AbstractElemShape, ApproximationType<:AbstractDerivativeOperator}
  @nospecialize rd
  print(io, "RefElemData{", summary(rd.approximationType), ", ", rd.elementType, "}")
end
