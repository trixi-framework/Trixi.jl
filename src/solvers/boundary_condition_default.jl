"""
    boundary_condition_default(mesh::AbstractMesh{1}, boundary_condition)

Create a default boundary conditions for a 1D mesh.
that uses the standard boundary naming convention.
This function applies the same boundary condition to all standard boundaries:
- `x_neg`: negative x-direction boundary
- `x_pos`: positive x-direction boundary

# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
- `mesh`: The [`StructuredMesh`](@ref) in 1D for which boundary conditions are created

# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default(::AbstractMesh{1}, boundary_condition)
    return (x_neg = boundary_condition,
            x_pos = boundary_condition)
end

"""
    boundary_condition_default(mesh::AbstractMesh{2}, boundary_condition)

Create default boundary conditions for a 2D mesh.
This function applies the same boundary condition to all standard boundaries:
- `x_neg`: negative x-direction boundary
- `x_pos`: positive x-direction boundary
- `y_neg`: negative y-direction boundary
- `y_pos`: positive y-direction boundary

# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
- `mesh`: The `mesh` in 2D for which boundary conditions are created

# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default(::AbstractMesh{2}, boundary_condition)
    return (x_neg = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition,
            x_pos = boundary_condition)
end

"""
    boundary_condition_default(mesh::AbstractMesh{3}, boundary_condition)

Create default boundary conditions for a 3D mesh.
This function applies the same boundary condition to all standard boundaries:
- `x_neg`: negative x-direction boundary
- `x_pos`: positive x-direction boundary
- `y_neg`: negative y-direction boundary
- `y_pos`: positive y-direction boundary
- `z_neg`: negative z-direction boundary
- `z_pos`: positive z-direction boundary

# Arguments
- `boundary_condition`: The boundary condition function to apply to all boundaries
- `mesh`: The `mesh` in 3D for which boundary conditions are created

# Returns
- Named tuple mapping boundary names to the boundary condition
"""
function boundary_condition_default(::AbstractMesh{3}, boundary_condition)
    return (x_neg = boundary_condition,
            x_pos = boundary_condition,
            y_neg = boundary_condition,
            y_pos = boundary_condition,
            z_neg = boundary_condition,
            z_pos = boundary_condition)
end
