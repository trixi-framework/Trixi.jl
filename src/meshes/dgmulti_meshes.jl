# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

"""
    DGMultiMesh{NDIMS, ...}

`DGMultiMesh` describes a mesh type which wraps `StartUpDG.MeshData` and `boundary_faces` in a
dispatchable type. This is intended to store geometric data and connectivities for any type of
mesh (Cartesian, affine, curved, structured/unstructured).
"""
struct DGMultiMesh{NDIMS, ElemType, MeshDataT <: MeshData{NDIMS}, BoundaryFaceT}
  md::MeshDataT
  boundary_faces::BoundaryFaceT
end

Base.ndims(::DGMultiMesh{NDIMS}) where {NDIMS} = NDIMS

function Base.show(io::IO, mesh::DGMultiMesh{NDIMS, ElemType}) where {NDIMS, ElemType}
  @nospecialize mesh # reduce precompilation time
  print(io, "$ElemType DGMultiMesh with NDIMS = $NDIMS.")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::DGMultiMesh{NDIMS, ElemType}) where {NDIMS, ElemType}
  @nospecialize mesh # reduce precompilation time
  if get(io, :compact, false)
    show(io, mesh)
  else
    summary_header(io, "DGMultiMesh{$NDIMS, $ElemType}, ")
    summary_line(io, "number of elements", mesh.md.num_elements)
    summary_line(io, "number of boundaries", length(mesh.boundary_faces))
    for (boundary_name, faces) in mesh.boundary_faces
      summary_line(increment_indent(io), "nfaces on $boundary_name", length(faces))
    end
    summary_footer(io)
  end
end

"""
    DGMultiMesh(vertex_coordinates::NTuple{NDIMS, Vector{Tv}}, EToV, rd::RefElemData;
                is_on_boundary = nothing,
                periodicity::NTuple{NDIMS, Bool} = ntuple(_->false, NDIMS)) where {NDIMS, Tv}

- `vertex_coordinates` is a tuple of vectors containing x,y,... components of the vertex coordinates
- `EToV` is a 2D array containing element-to-vertex connectivities for each element
- `rd` is a `RefElemData` from `StartUpDG.jl`, and contains information associated with to the
  reference element (e.g., quadrature, basis evaluation, differentiation, etc).
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `periodicity` is a tuple of booleans specifying if the domain is periodic `true`/`false` in the
   (x,y,z) direction.
"""
function DGMultiMesh(vertex_coordinates::NTuple{NDIMS, Vector{Tv}}, EToV::Array{Ti,2}, rd::RefElemData;
                     is_on_boundary = nothing,
                     periodicity=ntuple(_->false, NDIMS), kwargs...) where {NDIMS, Tv, Ti}

  if haskey(kwargs, :is_periodic)
    # TODO: DGMulti, v0.5. Remove deprecated keyword
    Base.depwarn("keyword argument `is_periodic` is now `periodicity`.", :DGMultiMesh)
    periodicity=kwargs[:is_periodic]
  end

  md = MeshData(vertex_coordinates, EToV, rd)
  md = StartUpDG.make_periodic(md, periodicity)
  boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
  return DGMultiMesh{NDIMS, typeof(rd.elementType), typeof(md), typeof(boundary_faces)}(md, boundary_faces)
end

# specialization for NDIMS = 1
function DGMultiMesh(vertex_coordinates::NTuple{1, Vector{Tv}}, EToV::Array{Ti,2}, rd::RefElemData;
                     is_on_boundary = nothing,
                     periodicity=(false, ), kwargs...) where {Tv, Ti}

  if haskey(kwargs, :is_periodic)
    # TODO: DGMulti, v0.5. Remove deprecated keyword
    Base.depwarn("keyword argument `is_periodic` is now `periodicity`.", :DGMultiMesh)
    periodicity=kwargs[:is_periodic]
  end

  md = MeshData(vertex_coordinates, EToV, rd)
  md = StartUpDG.make_periodic(md, periodicity...)
  boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
  return DGMultiMesh{1, typeof(rd.elementType), typeof(md), typeof(boundary_faces)}(md, boundary_faces)
end

"""
    DGMultiMesh(triangulateIO, rd::RefElemData{2, Tri}, boundary_dict::Dict{Symbol, Int})

- `triangulateIO` is a `TriangulateIO` mesh representation
- `rd` is a `RefElemData` from `StartUpDG.jl`, and contains information associated with to the
reference element (e.g., quadrature, basis evaluation, differentiation, etc).
- `boundary_dict` is a `Dict{Symbol, Int}` which associates each integer `TriangulateIO` boundary tag with a Symbol
"""
function DGMultiMesh(triangulateIO, rd::RefElemData{2, Tri}, boundary_dict::Dict{Symbol, Int})

  vertex_coordinates, EToV = StartUpDG.triangulateIO_to_VXYEToV(triangulateIO)
  md = MeshData(vertex_coordinates, EToV, rd)
  boundary_faces = StartUpDG.tag_boundary_faces(triangulateIO, rd, md, boundary_dict)
  return DGMultiMesh{2, typeof(rd.elementType), typeof(md), typeof(boundary_faces)}(md, boundary_faces)
end

# TODO: DGMulti, v0.5. Remove deprecated constructor
@deprecate VertexMappedMesh(args...; kwargs...) DGMultiMesh(args...; kwargs...)

end # @muladd
