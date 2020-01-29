module MeshMod

using ..Jul1dge

export Mesh


struct Mesh
  x_start::Real
  x_end::Real
  ncells::Integer
  coordinate::Array{Float64, 1}
  length::Array{Float64, 1}
end


function Mesh(x_start::Real, x_end::Real, ncells::Integer)
  mesh = Mesh(x_start, x_end, ncells, Array{Float64,1}(undef, ncells),
              Array{Float64,1}(undef, ncells))

  dx = (x_end - x_start) / ncells
  for c in 1:ncells
    mesh.coordinate[c] = x_start + dx/2 + (c - 1) * dx
    mesh.length[c] = dx
  end

  return mesh
end


end
