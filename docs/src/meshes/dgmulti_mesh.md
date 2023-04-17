# [Unstructured meshes and the `DGMulti` solver](@id DGMulti)

Trixi.jl includes support for simplicial and tensor product meshes via the `DGMulti` solver type,
which is based on the [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl) package.
`DGMulti` solvers also provide support for quadrilateral and hexahedral meshes, though this
feature is currently restricted to Cartesian grids.
On these line/quad/hex meshes, the `DGMulti` solver also allows to use all (finite domain) SBP
derivative operators provided by
[SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl),
including several finite difference SBP methods.

We make a few simplifying assumptions about supported meshes:
* meshes consist of a single type of element
* meshes are _conforming_ (e.g., each face of an element is shared with at most one other element).
* the geometric mapping from reference to physical elements is polynomial (currently, only affine
  mappings are supported).

`StartUpDG.jl` includes both simple uniform meshes via `uniform_mesh`, as well as support for triangular
meshes constructed using [Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl), a wrapper
around Jonathan Shewchuk's [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) package.

## The `DGMulti` solver type

Trixi.jl solvers on simplicial meshes use the `[DGMulti](@ref)` solver type, which allows users to specify
`element_type` and `approximation_type` in addition to `polydeg`, `surface_flux`, `surface_integral`,
and `volume_integral`.

```julia
DGMulti(; polydeg::Integer,
          element_type::AbstractElemShape,
          approximation_type=Polynomial(),
          surface_flux=flux_central,
          surface_integral=SurfaceIntegralWeakForm(surface_flux),
          volume_integral=VolumeIntegralWeakForm(),
          RefElemData_kwargs...)
```

Here, `element_type` can be `Tri()`, `Quad()`, `Tet()`, or `Hex()`, and `approximation_type` can be
* `Polynomial()`, which specifies a DG discretization using a polynomial basis using quadrature rules
  which are exact for degree `2 * polydeg` integrands, or
* `SBP()`, which specifies a DG discretization using multi-dimensional SBP operators. Types of SBP
  discretizations available include:
  `SBP{Kubatko{LobattoFaceNodes}}()` (the default choice), `SBP{Kubatko{LegendreFaceNodes}}()`, and
  [`SBP{Hicken}()`](https://doi.org/10.1007/s10915-020-01154-8). For `polydeg = 1, ..., 4`, the
  `SBP{Kubatko{LegendreFaceNodes}}()` SBP nodes are identical to the SBP nodes of
  [Chen and Shu](https://doi.org/10.1016/j.jcp.2017.05.025).
  More detailed descriptions of each SBP node set can be found in the
  [StartUpDG.jl docs](https://jlchan.github.io/StartUpDG.jl/dev/RefElemData/#RefElemData-based-on-SBP-finite-differences).
  Trixi.jl will also specialize certain parts of the solver based on the `SBP` approximation type.
* a (periodic or non-periodic) derivative operator from
  [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl),
  usually constructed as `D = derivative_operator(...)`. In this case, you do not need to
  pass a `polydeg`. Periodic derivative operators will only work with single-element meshes
  constructed using [`DGMultiMesh`](@ref).

Additional options can also be specified through `RefElemData_kwargs`:

* `quad_rule_vol = quad_nodes(Tri(), Nq)` will substitute in a volume quadrature rule of degree `Nq`
  instead of the default (which is a quadrature rule of degree `polydeg`).
  Here, a degree `Nq` rule will be exact for at least degree `2*Nq` integrands (such that the mass
  matrix is integrated exactly). Quadrature rules of which exactly integrate degree `Nq` integrands
  may also be specified (for example, `quad_rule_vol = StartUpDG.quad_nodes_tri(Nq)` on triangles).
* `quad_rule_face = quad_nodes(Line(), Nq))` will use a face quadrature rule of degree `Nq` rather
  than the default. This rule is also exact for at least degree `2*Nq` integrands.

### The `GaussSBP()` approximation type on `Quad()` and `Hex()` meshes

When using `VolumeIntegralFluxDifferencing` on `Quad()` and `Hex()` meshes, one can also specify
`approximation_type = GaussSBP()` to use an [entropy stable Gauss collocation scheme](https://doi.org/10.1137/18M1209234).
Here, `GaussSBP()` refers to "generalized" summation-by-parts operators (see for example
[Ranocha 2018](https://doi.org/10.1016/j.jcp.2018.02.021) or
[Fernandez and Zingg 2015](https://doi.org/10.1137/140992205)).

Unlike traditional SBP operators, generalized SBP operators are constructed from nodes which do
not include boundary nodes (i.e., Gauss quadrature nodes as opposed to Gauss-Lobatto quadrature
nodes). This makes the computation of interface fluxes slightly more expensive, but also usually
results in a more accurate solution. Roughly speaking, an entropy stable Gauss collocation scheme
will yield results similar to a modal entropy stable scheme using a `Polynomial()` approximation
type, but will be more efficient at high orders of approximation.

## Trixi.jl elixirs on simplicial and tensor product element meshes

Example elixirs with triangular, quadrilateral, and tetrahedral meshes can be found in
the `examples/dgmulti_2d` and `examples/dgmulti_3d` folders. Some key elixirs to look at:

* `examples/dgmulti_2d/elixir_euler_weakform.jl`: basic weak form DG discretization on a uniform triangular mesh.
  Changing `element_type = Quad()` or `approximation_type = SBP()` will switch to a quadrilateral mesh
  or an SBP-type discretization. Changing `surface_integral = SurfaceIntegralWeakForm(flux_ec)` and
  `volume_integral = VolumeIntegralFluxDifferencing(flux_ec)` for some entropy conservative flux
  (e.g., [`flux_chandrashekar`](@ref) or [`flux_ranocha`](@ref)) will switch to an entropy conservative formulation.
* `examples/dgmulti_2d/elixir_euler_triangulate_pkg_mesh.jl`: uses an unstructured mesh generated by
  [Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl).
* `examples/dgmulti_3d/elixir_euler_weakform.jl`: basic weak form DG discretization on a uniform tetrahedral mesh.
  Changing `element_type = Hex()` will switch to a hexahedral mesh. Changing
  `surface_integral = SurfaceIntegralWeakForm(flux_ec)` and
  `volume_integral = VolumeIntegralFluxDifferencing(flux_ec)` for some entropy conservative flux
  (e.g., [`flux_chandrashekar`](@ref) or [`flux_ranocha`](@ref)) will switch to an entropy conservative formulation.

# For developers

## `DGMultiMesh` wrapper type

`DGMulti` meshes in Trixi.jl are represented using a `DGMultiMesh{NDIMS, ...}` type.
This mesh type is assumed to have fields `md::MeshData`, which contains geometric terms
derived from the mapping between the reference and physical elements, and `boundary_faces`, which
contains a `Dict` of boundary segment names (symbols) and list of faces which lie on that boundary
segment.

A [`DGMultiMesh`](@ref) can be constructed in several ways. For example, `DGMultiMesh(dg::DGMulti)` will
return a Cartesian mesh on ``[-1, 1]^d`` with element types specified by `dg`.
`DGMulti` meshes can also be constructed by specifying a list
of vertex coordinates `vertex_coordinates_x`, `vertex_coordinates_y`, `vertex_coordinates_z` and a
connectivity matrix `EToV` where `EToV[e,:]` gives the vertices which correspond to element `e`.
These quantities are available from most unstructured mesh generators.

Initial support for curved `DGMultiMesh`es is available for flux differencing solvers using
`SBP` and `GaussSBP` approximation types on quadrilateral and hexahedral meshes. These can be
called by specifying `mesh = DGMultiMesh(dg, cells_per_dimension, mapping)`, where `mapping` is a
function which specifies the warping of the mesh (e.g., `mapping(xi, eta) = SVector{2}(xi, eta)` is
the identity mapping) similar to the `mapping` argument used by `StructuredMesh`.

## Variable naming conventions

We use the convention that coordinates on the reference element are ``r`` in 1D, ``r, s`` in 2D,
or ``r, s, t`` in 3D. Physical coordinates use the standard conventions ``x`` (1D),
``x, y`` (2D), and ``x, y, z`` (3D).

!["Ref-to-physical mapping"](https://user-images.githubusercontent.com/1156048/124361389-a2841380-dbf4-11eb-8ee4-33e71109c8bb.png)

Derivatives of reference coordinates with respect to physical coordinates are abbreviated, e.g.,
``\frac{\partial r}{\partial x} = r_x``. Additionally, ``J`` is used to denote the determinant of
the Jacobian of the reference-to-physical mapping.

## Variable meanings and conventions in `StartUpDG.jl`

`StartUpDG.jl` exports structs `RefElemData{NDIMS, ElemShape, ...}` (which contains data associated
with the reference element, such as interpolation points, quadrature rules, face nodes, normals,
and differentiation/interpolation/projection matrices) and `MeshData{NDIMS}` (which contains geometric
data associated with a mesh). These are currently used for evaluating DG formulations in a matrix-free
fashion. These structs contain fields similar (but not identical) to those in
`Globals1D, Globals2D, Globals3D` in the Matlab codes from "Nodal Discontinuous Galerkin Methods"
by [Hesthaven and Warburton (2007)](https://doi.org/10.1007/978-0-387-72067-8).

In general, we use the following code conventions:
* variables such as `r, s,...` and `x, y,...` correspond to values at nodal interpolation points.
* variables ending in `q` (e.g., `rq, sq,...` and `xq, yq,...`) correspond to values at volume
  quadrature points.
* variables ending in `f` (e.g., `rf, sf,...` and `xf, yf,...`) correspond to values at face
  quadrature points.
* variables ending in `p` (e.g., `rp, sp,...`) correspond to "plotting" points, which are usually
  a fine grid of equispaced points.
* `V` matrices correspond to interpolation matrices from nodal interpolation points, e.g., `Vq`
  interpolates to volume quadrature points, `Vf` interpolates to face quadrature points.
* geometric quantities in `MeshData` are stored as matrices of dimension
  ``\text{number of points per element} \times \text{number of elements}``.

Quantities in `rd::RefElemData`:
* `rd.Np, rd.Nq, rd.Nf`: the number of nodal interpolation points, volume quadrature points, and
  face quadrature points on the reference element, respectively.
* `rd.Vq`: interpolation matrices from values at nodal interpolation points to volume quadrature points
* `rd.wq`: volume quadrature weights on the reference element
* `rd.Vf`: interpolation matrices from values at nodal interpolation points to face quadrature points
* `rd.wf`: a vector containing face quadrature weights on the reference element
* `rd.M`: the quadrature-based mass matrix, computed via `rd.Vq' * diagm(rd.wq) * rd.Vq`.
* `rd.Pq`: a quadrature-based ``L^2`` projection matrix `rd.Pq = rd.M \ rd.Vq' * diagm(rd.wq)`
  which maps between values at quadrature points and values at nodal points.
* `Dr, Ds, Dt` matrices are nodal differentiation matrices with respect to the ``r,s,t`` coordinates,
  e.g., `Dr*f.(r,s)` approximates the derivative of ``f(r,s)`` at nodal points.

Quantities in `md::MeshData`:
* `md.xyz` is a tuple of matrices `md.x`, `md.y`, `md.z`, where column `e` contains coordinates of
  physical interpolation points.
* `md.xyzq` is a tuple of matrices `md.xq`, `md.yq`, `md.zq`, where column `e` contains coordinates
  of physical quadrature points.
* `md.rxJ, md.sxJ, ...` are matrices where column `e` contains values of
  ``J\frac{\partial r}{\partial x}``, ``J\frac{\partial s}{\partial x}``, etc. at nodal interpolation
  points on the element `e`.
* `md.J` is a matrix where column `e` contains values of the Jacobian ``J`` at nodal interpolation points.
* `md.Jf` is a matrix where column `e` contains values of the face Jacobian (e.g., determinant of
  the geometric mapping between a physical face and a reference face) at face quadrature points.
* `md.nxJ, md.nyJ, ...` are matrices where column `e` contains values of components of the unit
  normal scaled by the face Jacobian `md.Jf` at face quadrature points.

For more details, please see the [StartUpDG.jl docs](https://jlchan.github.io/StartUpDG.jl/dev/).
