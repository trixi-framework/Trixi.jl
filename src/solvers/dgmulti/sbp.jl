
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
  Dr = droptol!(sparse(D), tol)
  M = Diagonal(wq)
  Pq = LinearAlgebra.I
  Vq = LinearAlgebra.I

  VDM = nothing # TODO: generalized Vandermonde matrix

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
