
"""
    DGMulti(approximation_type::AbstractDerivativeOperator;
            element_type::AbstractElemShape,
            surface_flux=flux_central,
            surface_integral=SurfaceIntegralWeakForm(surface_flux),
            volume_integral=VolumeIntegralWeakForm(),
            kwargs...)

Create a summation by parts (SBP) discretization on the given `element_type`
using a tensor product structure based on the 1D SBP derivative operator
passed as `approximation_type`.

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
    # TODO: DGMulti. Remove this branch and use the non-periodic code for all
    #                operators.
    # Periodic operators do not include both boundary nodes in their
    # computational grid. Thus, they cannot be handled in the same way as
    # non-periodic operators.
    # Currently, we only support periodic operators with our special
    # `CartesianMesh` constructor, which gets the geometry information from
    # the DGMulti solver itself. Hence, the nodes of the mesh will always be
    # the same as the nodes of the solver and we do not need to adjust anything.
    factor = one(eltype(nodes_1d))
  else
    # We can adjust the grid spacing as follows.
    xmin = SummationByPartsOperators.xmin(D)
    xmax = SummationByPartsOperators.xmax(D)
    factor = 2 / (xmax - xmin)
    @. nodes_1d = factor * (nodes_1d - xmin) - 1
    @. weights_1d = factor * weights_1d
  end

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
  if D isa AbstractPeriodicDerivativeOperator
    # we do not need any face stuff for periodic operators
    Vf = spzeros(length(wf), length(wq))
  else
    Vf = sparse([1, 2], [1, length(nodes_1d)], [1.0, 1.0])
  end
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
  face_mask = vcat(StartUpDG.find_face_nodes(element_type, r, s)...)

  rf, sf, wf, nrJ, nsJ = StartUpDG.init_face_data(element_type,
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
    StartUpDG.init_face_data(element_type, quad_rule_face=(rf, sf, wf))
  end
  if D isa AbstractPeriodicDerivativeOperator
    # we do not need any face stuff for periodic operators
    Vf = spzeros(length(wf), length(wq))
  else
    Vf = sparse(eachindex(face_mask), face_mask, ones(Bool, length(face_mask)))
  end
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

function StartUpDG.inverse_trace_constant(rd::RefElemData{NDIMS, ElementType, ApproximationType})  where {NDIMS, ElementType<:Union{Line, Quad, Hex}, ApproximationType<:AbstractDerivativeOperator}
  D = rd.approximationType

  # the inverse trace constant is the maximum eigenvalue corresponding to
  #       M_f * v = λ * M * v
  # where M_f is the face mass matrix and M is the volume mass matrix.
  # Since M is diagonal and since M_f is just the boundary "mask" matrix
  # (which extracts the first and last entries of a vector), the maximum
  # eigenvalue is the inverse of the first or last mass matrix diagonal.
  left_weight = SummationByPartsOperators.left_boundary_weight(D)
  right_weight = SummationByPartsOperators.right_boundary_weight(D)
  max_eigenvalue = max(inv(left_weight), inv(right_weight))

  # For tensor product elements, the trace constant for higher dimensional
  # elements is the one-dimensional trace constant multiplied by `NDIMS`. See
  #     "GPU-accelerated discontinuous Galerkin methods on hybrid meshes."
  #     Chan, Jesse, et al (2016), https://doi.org/10.1016/j.jcp.2016.04.003
  # for more details (specifically, Appendix A.1, Theorem A.4).
  return NDIMS * max_eigenvalue
end

# type alias for specializing on a periodic SBP operator
const DGMultiPeriodicFDSBP{NDIMS, ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} where {NDIMS, ElemType, ApproxType<:SummationByPartsOperators.AbstractPeriodicDerivativeOperator, SurfaceIntegral, VolumeIntegral}

"""
    CartesianMesh(dg::DGMulti)

Constructs a single-element [`VertexMappedMesh`](@ref) for a single periodic element given
a DGMulti with `approximation_type` set to a periodic (finite difference) SBP operator from
SummationByPartsOperators.jl.
"""
function CartesianMesh(dg::DGMultiPeriodicFDSBP{NDIMS};
                       coordinates_min=ntuple(_ -> -1.0, NDIMS),
                       coordinates_max=ntuple(_ -> 1.0, NDIMS)) where {NDIMS}

  rd = dg.basis

  e = ones(size(rd.r))
  z = zero(e)

  VXYZ = ntuple(_ -> [], NDIMS)
  EToV = NaN # StartUpDG.jl uses size(EToV, 1) for the number of elements, this lets us reuse that.
  FToF = []

  xyz = xyzq = rd.rst
  xyzf = ntuple(_ -> [], NDIMS)
  wJq = diag(rd.M)

  # arrays of connectivity indices between face nodes
  mapM = mapP = mapB = []

  # volume geofacs Gij = dx_i/dxhat_j
  coord_diffs = coordinates_max .- coordinates_min

  # Periodic SBP operators do not include one endpoint. We account for this by adding
  # `h`` when estimating `factor`, which is the size of the "reference interval".
  D = rd.approximationType
  x = grid(D)
  h = x[2] - x[1]
  factor = (x -> x[2] - x[1])(extrema(grid(D))) + h

  if NDIMS==1
    rxJ = coord_diffs[1] / factor
    rstxyzJ = @SMatrix [rxJ * e]
  elseif NDIMS==2
    rxJ = coord_diffs[1] / factor
    syJ = coord_diffs[2] / factor
    rstxyzJ = @SMatrix [rxJ * e z; z syJ * e]
  elseif NDIMS==3
    rxJ = coord_diffs[1] / factor
    syJ = coord_diffs[2] / factor
    tzJ = coord_diffs[3] / factor
    rstxyzJ = @SMatrix [rxJ * e z z; z syJ * e z; z z tzJ * e]
  end

  J = e * prod(coord_diffs) / factor^NDIMS

  # surface geofacs
  nxyzJ = ntuple(_ -> [], NDIMS)
  Jf = []

  is_periodic = ntuple(_ -> true, NDIMS)

  md = MeshData(VXYZ, EToV, FToF, xyz, xyzf, xyzq, wJq,
                mapM, mapP, mapB, rstxyzJ, J, nxyzJ, Jf,
                is_periodic)

  boundary_faces = []
  n_boundary_faces = length(boundary_faces)
  return VertexMappedMesh{NDIMS, rd.elementType, typeof(md), n_boundary_faces, typeof(boundary_faces)}(md, boundary_faces)
end

# This is used in `estimate_dt`. `estimate_h` uses that `Jf / J = O(h^{NDIMS-1}) / O(h^{NDIMS}) = O(1/h)`.
# However, since we do not initialize `Jf` for periodic FDSBP operators, we specialize `estimate_h`
# based on the grid provided by SummationByPartsOperators.jl.
function StartUpDG.estimate_h(e, rd::RefElemData{NDIMS, ElementType, ApproximationType}, md::MeshData)  where {NDIMS, ElementType<:StartUpDG.AbstractElemShape, ApproximationType<:SummationByPartsOperators.AbstractPeriodicDerivativeOperator}
  D = rd.approximationType
  x = grid(D)
  return x[2] - x[1]
end

# specialized for DGMultiPeriodicFDSBP since there are no face nodes
# and thus no inverse trace constant for periodic domains.
function estimate_dt(mesh::AbstractMeshData, dg::DGMultiPeriodicFDSBP)
  rd = dg.basis # RefElemData
  return StartUpDG.estimate_h(rd, mesh.md)
end

# do nothing for interface terms if using a periodic operator
function prolong2interfaces!(cache, u, mesh::AbstractMeshData, equations,
                             surface_integral, dg::DGMultiPeriodicFDSBP)
  @assert nelements(mesh, dg, cache) == 1
  nothing
end

function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm,
                              mesh::VertexMappedMesh,
                              have_nonconservative_terms::Val{false}, equations,
                              dg::DGMultiPeriodicFDSBP)
  @assert nelements(mesh, dg, cache) == 1
  nothing
end

function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm,
                                mesh::VertexMappedMesh, equations,
                                dg::DGMultiPeriodicFDSBP, cache)
  @assert nelements(mesh, dg, cache) == 1
  nothing
end
