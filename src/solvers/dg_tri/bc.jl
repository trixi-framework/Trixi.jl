struct BoundaryStateDirichlet{B}
  boundary_value_function::B
end

# Dirichlet-type boundary condition for use with TreeMesh or CurvedMesh
@inline function (boundary_condition::BoundaryStateDirichlet)(u_inner, normal_direction,
                                                              x, t, equations)
  return boundary_condition.boundary_value_function(x, t, equations)
end

struct BoundaryStateWall{B}
  boundary_value_function::B
end

@inline function (boundary_condition::BoundaryStateWall)(u_inner, normal_direction::AbstractVector,
                                                         x, t, equations)
  return boundary_condition.boundary_value_function(u_inner, normal_direction, equations)
end

# interface with semidiscretization_hyperbolic
function Trixi.digest_boundary_conditions(boundary_conditions::NamedTuple{Keys,ValueTypes}, 
                                          mesh::AbstractMeshData, dg::DG{<:RefElemData}, cache) where {Keys,ValueTypes<:Tuple{Any,Any}}
    return boundary_conditions
end