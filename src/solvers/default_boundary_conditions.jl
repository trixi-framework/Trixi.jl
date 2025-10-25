"""
    boundary_condition_default(mesh::P4estMesh{2,2}, boundary_condition)

Create a default boundary condition dictionary for p4est meshes in 2D
that use the standard boundary naming convention.

This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary

# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries

# Returns
- `Dict{Symbol, Any}`: Dictionary mapping boundary names to the boundary condition
"""
function  boundary_condition_default(mesh::P4estMesh{2,2}, boundary_condition)

     return Dict(:x_neg => boundary_condition,
                 :y_neg => boundary_condition,
                 :y_pos => boundary_condition,
                 :x_pos => boundary_condition)
end 

"""
    boundary_condition_default(mesh::P4estMesh{3,3}, boundary_condition)

Create a default boundary condition dictionary for p4est meshes in 3D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
- `:z_neg`: negative z-direction boundary
- `:z_pos`: positive z-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- `Dict{Symbol, Any}`: Dictionary mapping boundary names to the boundary condition
"""
function boundary_condition_default(mesh::P4estMesh{3,3}, boundary_condition)

     return Dict(:x_neg => boundary_condition,
                  :x_pos => boundary_condition,
                  :y_neg => boundary_condition,
                  :y_pos => boundary_condition,
                  :z_neg => boundary_condition,
                  :z_pos => boundary_condition)
end

"""
    boundary_condition_default(mesh::StructuredMesh1D, boundary_condition)

Create a default boundary condition dictionary for structured meshes in 1D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary   
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default(mesh::StructuredMesh{1}, boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition)
end

"""
    boundary_condition_default(mesh::StructuredMesh2D, boundary_condition)

Create a default boundary condition dictionary for structured meshes in 2D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary   
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function  boundary_condition_default(mesh::StructuredMesh{2}, boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition)
end

"""
    boundary_condition_default(mesh::StructuredMesh3D, boundary_condition)

Create a default boundary condition dictionary for structured meshes in 3D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary  
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
- `:z_neg`: negative z-direction boundary
- `:z_pos`: positive z-direction boundary 
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function  boundary_condition_default(mesh::StructuredMesh{3}, boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition,
            z_neg = boundary_condition,
            z_pos = boundary_condition)

end 

"""
    boundary_condition_default(mesh::TreeMesh1D, boundary_condition)

Create a default boundary condition dictionary for tree meshes in 1D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default(mesh::TreeMesh1D, boundary_condition)
    return (x_neg = boundary_condition,
            x_pos = boundary_condition)
end

"""
    boundary_condition_default(mesh::TreeMesh2D, boundary_condition)

Create a default boundary condition dictionary for tree meshes in 2D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary    
- `:x_pos`: positive x-direction boundary
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function  boundary_condition_default(mesh::TreeMesh2D, boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition)
end

"""
   boundary_condition_default(mesh::TreeMesh3D, boundary_condition)

Create a default boundary condition dictionary for tree meshes in 3D
that use the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `:x_neg`: negative x-direction boundary
- `:x_pos`: positive x-direction boundary
- `:y_neg`: negative y-direction boundary
- `:y_pos`: positive y-direction boundary
- `:z_neg`: negative z-direction boundary
- `:z_pos`: positive z-direction boundary
# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function  boundary_condition_default(mesh::TreeMesh3D, boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition,
            z_neg = boundary_condition,
            z_pos = boundary_condition)
end