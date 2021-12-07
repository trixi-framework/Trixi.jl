
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



function construct_1d_operators(D::AbstractDerivativeOperator, tol)
  nodes_1d = collect(grid(D))
  M = SummationByPartsOperators.mass_matrix(D)
  if M isa UniformScaling
    weights_1d = M * ones(Bool, length(nodes_1d))
  else
    weights_1d = diag(M)
  end

  # StartUpDG assumes nodes from -1 to +1. Thus, we need to re-scale everything.
  if D isa AbstractPeriodicDerivativeOperator
    # Periodic operators do not include both boundary nodes in their
    # computational grid. Thus, we need to use their "evaluation grid"
    # including both boundary nodes to determine the grid spacing factor.
    xmin, xmax = extrema(D.grid_evaluate)
  else
    xmin, xmax = extrema(nodes_1d)
  end
  factor = 2 / (xmax - xmin)
  @. nodes_1d = factor * (nodes_1d - xmin) - 1
  @. weights_1d = factor * weights_1d

  D_1d = droptol!(inv(factor) * sparse(D), tol)
  I_1d = Diagonal(ones(Bool, length(nodes_1d)))

  return nodes_1d, weights_1d, D_1d, I_1d
end


function StartUpDG.RefElemData(element_type::Line,
                               D::AbstractDerivativeOperator;
                               tol = 100*eps())

  approximation_type = D
  N = SummationByPartsOperators.accuracy_order(D) # kind of polynomial degree

  # 1D operators
  nodes_1d, weights_1d, D_1d = construct_1d_operators(D, tol)

  # volume
  rq = r = nodes_1d
  wq = weights_1d
  Dr = D_1d
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

  # 1D operators
  nodes_1d, weights_1d, D_1d, I_1d = construct_1d_operators(D, tol)

  # volume
  s, r = vec.(StartUpDG.NodesAndModes.meshgrid(nodes_1d)) # this is to match
                                                          # ordering of nrstJ
  rq = r; sq = s
  wr, ws = vec.(StartUpDG.NodesAndModes.meshgrid(weights_1d))
  wq = wr .* ws
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
  # if D isa AbstractPeriodicDerivativeOperator
  #   # we need to fake the face mask for periodic operators for StartUpDG to work
  #   xmin, xmax = extrema(r)
  #   factor = 2 / (xmax - xmin)
  #   face_mask = vcat(StartUpDG.find_face_nodes(element_type,
  #     @.(factor * (r - xmin) - 1),
  #     @.(factor * (s - xmin) - 1))...)
  # else
    face_mask = vcat(StartUpDG.find_face_nodes(element_type, r, s)...)
  # end

  rf, sf, wf, nrJ, nsJ = StartUpDG.init_face_data(element_type, N,
    quad_rule_face=(nodes_1d, weights_1d))
  if D isa AbstractPeriodicDerivativeOperator
    # we do not need any face stuff for periodic operators
    Vf = spzeros(length(wf), length(wq))
  else
    Vf = sparse(eachindex(face_mask), face_mask, ones(Bool, length(face_mask)))
  end
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

  # 1D operators
  nodes_1d, weights_1d, D_1d, I_1d = construct_1d_operators(D, tol)

  # volume
  # to match ordering of nrstJ
  s, r, t = vec.(StartUpDG.NodesAndModes.meshgrid(nodes_1d, nodes_1d, nodes_1d))
  rq = r; sq = s; tq = t
  wr, ws, wt = vec.(StartUpDG.NodesAndModes.meshgrid(weights_1d, weights_1d, weights_1d))
  wq = wr .* ws .* wt
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



# specializations for periodic operators
function prolong2interfaces!(
    cache, u, mesh::AbstractMeshData, equations, surface_integral,
    dg::DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral}) where {NDIMS, ElemType, ApproxType<:AbstractPeriodicDerivativeOperator, SurfaceIntegral, VolumeIntegral}

  @assert nelements(mesh, dg, cache) == 1
end

function calc_interface_flux!(
    cache, surface_integral::SurfaceIntegralWeakForm, mesh::VertexMappedMesh,
    nonconservative_terms::Val{true}, equations,
    dg::DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral}) where {NDIMS, ElemType, ApproxType<:AbstractPeriodicDerivativeOperator, SurfaceIntegral, VolumeIntegral}

  @assert nelements(mesh, dg, cache) == 1
end
function calc_interface_flux!(
    cache, surface_integral::SurfaceIntegralWeakForm, mesh::VertexMappedMesh,
    nonconservative_terms::Val{false}, equations,
    dg::DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral}) where {NDIMS, ElemType, ApproxType<:AbstractPeriodicDerivativeOperator, SurfaceIntegral, VolumeIntegral}

  @assert nelements(mesh, dg, cache) == 1
end

function calc_boundary_flux!(
    cache, t, boundary_conditions::BoundaryConditionPeriodic, mesh, equations,
    dg::DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral}) where {NDIMS, ElemType, ApproxType<:AbstractPeriodicDerivativeOperator, SurfaceIntegral, VolumeIntegral}

  @assert sum(length, values(mesh.boundary_faces)) == 0
end
function calc_boundary_flux!(
    cache, t, boundary_conditions, mesh, equations,
    dg::DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral}) where {NDIMS, ElemType, ApproxType<:AbstractPeriodicDerivativeOperator, SurfaceIntegral, VolumeIntegral}

  @assert sum(length, values(mesh.boundary_faces)) == 0
end

function calc_surface_integral!(
    du, u, surface_integral::SurfaceIntegralWeakForm,
    mesh::VertexMappedMesh, equations,
    dg::DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral},
    cache) where {NDIMS, ElemType, ApproxType<:AbstractPeriodicDerivativeOperator, SurfaceIntegral, VolumeIntegral}

  @assert nelements(mesh, dg, cache) == 1
end
