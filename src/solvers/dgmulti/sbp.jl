
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
  # We can adjust the grid spacing as follows.
  xmin = SummationByPartsOperators.xmin(D)
  xmax = SummationByPartsOperators.xmax(D)
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
    rst, LinearAlgebra.I, # plotting
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
    rst, LinearAlgebra.I, # plotting
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
    rst, LinearAlgebra.I, # plotting
    rstq, wq, Vq, # quadrature
    rstf, wf, Vf, nrstJ, # faces
    M, Pq, Drst, LIFT)
end

# specialized Hex constructor in 3D to reduce memory usage.
function StartUpDG.RefElemData(element_type::Hex,
                               D::AbstractPeriodicDerivativeOperator;
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
  # We do not need any face data for periodic operators. Thus, we just
  # pass `nothing` to save memory.
  face_vertices = ntuple(_ -> nothing, 3)
  face_mask = nothing
  wf = nothing
  rstf = ntuple(_ -> nothing, 3)
  nrstJ = ntuple(_ -> nothing, 3)
  Vf = nothing
  LIFT = nothing

  # low order interpolation nodes
  V1 = nothing # do not need to store V1, since we specialize StartUpDG.MeshData to avoid using it.

  return RefElemData(
    element_type, approximation_type, N,
    face_vertices, V1,
    rst, VDM, face_mask,
    rst, LinearAlgebra.I, # plotting
    rstq, wq, Vq, # quadrature
    rstf, wf, Vf, nrstJ, # faces
    M, Pq, Drst, LIFT)
end


function Base.show(io::IO, mime::MIME"text/plain", rd::RefElemData{NDIMS, ElementType, ApproximationType}) where {NDIMS, ElementType<:StartUpDG.AbstractElemShape, ApproximationType<:AbstractDerivativeOperator}
  @nospecialize rd
  print(io, "RefElemData for an approximation using an ")
  show(IOContext(io, :compact => true), rd.approximation_type)
  print(io, " on $(rd.element_type) element")
end

function Base.show(io::IO, rd::RefElemData{NDIMS, ElementType, ApproximationType}) where {NDIMS, ElementType<:StartUpDG.AbstractElemShape, ApproximationType<:AbstractDerivativeOperator}
  @nospecialize rd
  print(io, "RefElemData{", summary(rd.approximation_type), ", ", rd.element_type, "}")
end

function StartUpDG.inverse_trace_constant(rd::RefElemData{NDIMS, ElementType, ApproximationType})  where {NDIMS, ElementType<:Union{Line, Quad, Hex}, ApproximationType<:AbstractDerivativeOperator}
  D = rd.approximation_type

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

const DGMultiFluxDiffPeriodicFDSBP{NDIMS, ApproxType, ElemType} =
  DGMulti{NDIMS, ElemType, ApproxType, SurfaceIntegral, VolumeIntegral} where {NDIMS, ElemType, ApproxType<:SummationByPartsOperators.AbstractPeriodicDerivativeOperator, SurfaceIntegral<:SurfaceIntegralWeakForm, VolumeIntegral<:VolumeIntegralFluxDifferencing}

"""
    DGMultiMesh(dg::DGMulti)

Constructs a single-element [`DGMultiMesh`](@ref) for a single periodic element given
a DGMulti with `approximation_type` set to a periodic (finite difference) SBP operator from
SummationByPartsOperators.jl.
"""
function DGMultiMesh(dg::DGMultiPeriodicFDSBP{NDIMS};
                     coordinates_min=ntuple(_ -> -one(real(dg)), NDIMS),
                     coordinates_max=ntuple(_ -> one(real(dg)), NDIMS)) where {NDIMS}

  rd = dg.basis

  e = Ones{eltype(rd.r)}(size(rd.r))
  z = Zeros{eltype(rd.r)}(size(rd.r))

  VXYZ = ntuple(_ -> [], NDIMS)
  EToV = NaN # StartUpDG.jl uses size(EToV, 1) for the number of elements, this lets us reuse that.
  FToF = []

  # We need to scale the domain from `[-1, 1]^NDIMS` (default in StartUpDG.jl)
  # to the given `coordinates_min, coordinates_max`
  xyz = xyzq = map(copy, rd.rst)
  for dim in 1:NDIMS
    factor = (coordinates_max[dim] - coordinates_min[dim]) / 2
    @. xyz[dim] = factor * (xyz[dim] + 1) + coordinates_min[dim]
  end
  xyzf = ntuple(_ -> [], NDIMS)
  wJq = diag(rd.M)

  # arrays of connectivity indices between face nodes
  mapM = mapP = mapB = []

  # volume geofacs Gij = dx_i/dxhat_j
  coord_diffs = coordinates_max .- coordinates_min

  J_scalar = prod(coord_diffs) / 2^NDIMS
  J = e * J_scalar

  if NDIMS == 1
    rxJ = J_scalar * 2 / coord_diffs[1]
    rstxyzJ = @SMatrix [rxJ * e]
  elseif NDIMS == 2
    rxJ = J_scalar * 2 / coord_diffs[1]
    syJ = J_scalar * 2 / coord_diffs[2]
    rstxyzJ = @SMatrix [rxJ * e z; z syJ * e]
  elseif NDIMS == 3
    rxJ = J_scalar * 2 / coord_diffs[1]
    syJ = J_scalar * 2 / coord_diffs[2]
    tzJ = J_scalar * 2 / coord_diffs[3]
    rstxyzJ = @SMatrix [rxJ * e z z; z syJ * e z; z z tzJ * e]
  end

  # surface geofacs
  nxyzJ = ntuple(_ -> [], NDIMS)
  Jf = []

  periodicity = ntuple(_ -> true, NDIMS)

  if NDIMS == 1
    mesh_type = Line()
  elseif NDIMS == 2
    mesh_type = Quad()
  elseif NDIMS == 3
    mesh_type = Hex()
  end

  md = MeshData(StartUpDG.VertexMappedMesh(mesh_type, VXYZ, EToV), FToF, xyz, xyzf, xyzq, wJq,
                mapM, mapP, mapB, rstxyzJ, J, nxyzJ, Jf,
                periodicity)

  boundary_faces = []
  return DGMultiMesh{NDIMS, rd.element_type, typeof(md), typeof(boundary_faces)}(md, boundary_faces)
end

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# This is used in `estimate_dt`. `estimate_h` uses that `Jf / J = O(h^{NDIMS-1}) / O(h^{NDIMS}) = O(1/h)`.
# However, since we do not initialize `Jf` for periodic FDSBP operators, we specialize `estimate_h`
# based on the reference grid provided by SummationByPartsOperators.jl and information about the domain size
# provided by `md::MeshData``.
function StartUpDG.estimate_h(e, rd::RefElemData{NDIMS, ElementType, ApproximationType}, md::MeshData)  where {NDIMS, ElementType<:StartUpDG.AbstractElemShape, ApproximationType<:SummationByPartsOperators.AbstractPeriodicDerivativeOperator}
  D = rd.approximation_type
  x = grid(D)

  # we assume all SummationByPartsOperators.jl reference grids are rescaled to [-1, 1]
  xmin = SummationByPartsOperators.xmin(D)
  xmax = SummationByPartsOperators.xmax(D)
  factor = 2 / (xmax - xmin)

  # If the domain has size L^NDIMS, then `minimum(md.J)^(1 / NDIMS) = L`.
  # WARNING: this is not a good estimate on anisotropic grids.
  return minimum(diff(x)) * factor * minimum(md.J)^(1 / NDIMS)
end

# specialized for DGMultiPeriodicFDSBP since there are no face nodes
# and thus no inverse trace constant for periodic domains.
function estimate_dt(mesh::DGMultiMesh, dg::DGMultiPeriodicFDSBP)
  rd = dg.basis # RefElemData
  return StartUpDG.estimate_h(rd, mesh.md)
end

# do nothing for interface terms if using a periodic operator
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache, u, mesh::DGMultiMesh, equations,
                             surface_integral, dg::DGMultiPeriodicFDSBP)
  @assert nelements(mesh, dg, cache) == 1
  nothing
end

function calc_interface_flux!(cache, surface_integral::SurfaceIntegralWeakForm,
                              mesh::DGMultiMesh,
                              have_nonconservative_terms::False, equations,
                              dg::DGMultiPeriodicFDSBP)
  @assert nelements(mesh, dg, cache) == 1
  nothing
end

function calc_surface_integral!(du, u, surface_integral::SurfaceIntegralWeakForm,
                                mesh::DGMultiMesh, equations,
                                dg::DGMultiPeriodicFDSBP, cache)
  @assert nelements(mesh, dg, cache) == 1
  nothing
end

function create_cache(mesh::DGMultiMesh, equations,
                      dg::DGMultiFluxDiffPeriodicFDSBP, RealT, uEltype)

  md = mesh.md

  # storage for volume quadrature values, face quadrature values, flux values
  nvars = nvariables(equations)
  u_values = allocate_nested_array(uEltype, nvars, size(md.xq), dg)
  return (; u_values, invJ = inv.(md.J) )
end

# Specialize calc_volume_integral for periodic SBP operators (assumes the operator is sparse).
function calc_volume_integral!(du, u, mesh::DGMultiMesh,
                               have_nonconservative_terms::False, equations,
                               volume_integral::VolumeIntegralFluxDifferencing,
                               dg::DGMultiFluxDiffPeriodicFDSBP, cache)

  @unpack volume_flux = volume_integral

  # We expect speedup over the serial version only when using two or more threads
  # since the threaded version below does not exploit the symmetry properties,
  # resulting in a performance penalty of 1/2
  if Threads.nthreads() > 1

    for dim in eachdim(mesh)
      normal_direction = get_contravariant_vector(1, dim, mesh, cache)

      # These are strong-form operators of the form `D = M \ Q` where `M` is diagonal
      # and `Q` is skew-symmetric. Since `M` is diagonal, `inv(M)` scales the rows of `Q`.
      # Then, `1 / M[i,i] * ∑_j Q[i,j] * volume_flux(u[i], u[j])` is equivalent to
      #       `= ∑_j (1 / M[i,i] * Q[i,j]) * volume_flux(u[i], u[j])`
      #       `= ∑_j        D[i,j]         * volume_flux(u[i], u[j])`
      # TODO: DGMulti.
      # This would have to be changed if `has_nonconservative_terms = False()`
      # because then `volume_flux` is non-symmetric.
      A = dg.basis.Drst[dim]

      A_base = parent(A) # the adjoint of a SparseMatrixCSC is basically a SparseMatrixCSR
      row_ids = axes(A, 2)
      rows = rowvals(A_base)
      vals = nonzeros(A_base)

      @threaded for i in row_ids
        u_i = u[i]
        du_i = du[i]
        for id in nzrange(A_base, i)
          j = rows[id]
          u_j = u[j]
          A_ij = vals[id]
          AF_ij = 2 * A_ij * volume_flux(u_i, u_j, normal_direction, equations)
          du_i = du_i + AF_ij
        end
        du[i] = du_i
      end
    end

  else # if using two threads or fewer

    # Calls `hadamard_sum!``, which uses symmetry to reduce flux evaluations. Symmetry
    # is expected to yield about a 2x speedup, so we default to the symmetry-exploiting
    # volume integral unless we have >2 threads (which should yield >2 speedup).
    for dim in eachdim(mesh)
      normal_direction = get_contravariant_vector(1, dim, mesh, cache)

      A = dg.basis.Drst[dim]

      # since has_nonconservative_terms::False,
      # the volume flux is symmetric.
      flux_is_symmetric = True()
      hadamard_sum!(du, A, flux_is_symmetric, volume_flux,
                    normal_direction, u, equations)
    end

  end
end

end # @muladd
