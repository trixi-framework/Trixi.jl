"""
    boundary_condition_default_p4est_2D(boundary_condition)

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
function boundary_condition_default_p4est_2D(boundary_condition)

     return Dict(:x_neg => boundary_condition,
                 :y_neg => boundary_condition,
                 :y_pos => boundary_condition,
                 :x_pos => boundary_condition)
end 

"""
    boundary_condition_default_p4est_3D(boundary_condition)

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
function boundary_condition_default_p4est_3D(boundary_condition)

     return Dict(:x_neg => boundary_condition,
                  :x_pos => boundary_condition,
                  :y_neg => boundary_condition,
                  :y_pos => boundary_condition,
                  :z_neg => boundary_condition,
                  :z_pos => boundary_condition)
end


"""
    boundary_condition_default_structured_1D(boundary_condition)

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
function boundary_condition_default_structured_1D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition)
end

"""
    boundary_condition_default_structured_2D(boundary_condition)

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
function boundary_condition_default_structured_2D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition)
end

"""
    boundary_condition_default_structured_3D(boundary_condition)

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
function boundary_condition_default_structured_3D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition,
            z_neg = boundary_condition,
            z_pos = boundary_condition)

end 

"""
    boundary_condition_default_tree_1D(boundary_condition)

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
function boundary_condition_default_tree_1D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition)
end

"""
    boundary_condition_default_tree_2D(boundary_condition)

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
function boundary_condition_default_tree_2D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition)
end

"""
    boundary_condition_default_tree_3D(boundary_condition)

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
function boundary_condition_default_tree_3D(boundary_condition)

    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition,
            z_neg = boundary_condition,
            z_pos = boundary_condition)
end

"""
    boundary_condition_default(mesh, boundary_condition)

Create a default boundary condition dictionary based on the mesh type.
This function automatically determines the appropriate boundary condition
format based on the mesh type and applies the given boundary condition
to all boundaries.

# Arguments
- `mesh`: The mesh object (TreeMesh, StructuredMesh, P4estMesh)
- `boundary_condition`: The boundary condition function to apply to all boundaries

# Returns
- Dictionary or named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default(mesh, boundary_condition)
    
    if isa(mesh, TreeMesh)
        ndims_mesh = ndims(mesh)
        if ndims_mesh == 1
            return boundary_condition_default_tree_1D(boundary_condition)
        elseif ndims_mesh == 2
            return boundary_condition_default_tree_2D(boundary_condition)
        elseif ndims_mesh == 3
            return boundary_condition_default_tree_3D(boundary_condition)
        end
    elseif isa(mesh, StructuredMesh)
        ndims_mesh = ndims(mesh)
        if ndims_mesh == 1
            return boundary_condition_default_structured_1D(boundary_condition)
        elseif ndims_mesh == 2
            return boundary_condition_default_structured_2D(boundary_condition)
        elseif ndims_mesh == 3
            return boundary_condition_default_structured_3D(boundary_condition)
        end
    elseif isa(mesh, P4estMesh)
        ndims_mesh = ndims(mesh)
        if ndims_mesh == 2
            return boundary_condition_default_p4est_2D(boundary_condition)
        elseif ndims_mesh == 3
            return boundary_condition_default_p4est_3D(boundary_condition)
        end
    
    end
    
    error("No default boundary conditions available for mesh type $(typeof(mesh)). " *
          "Please specify the boundary conditions manually.")
end