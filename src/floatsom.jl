#TODO: cleanup and remove old routines

# FROM equations.jl

# Wall boundary condition for use with TreeMesh or StructuredMesh
@inline function (boundary_condition::BoundaryConditionWall)(u_inner, orientation_or_normal,
                                                             direction,
                                                             x, t,
                                                             surface_flux_function, equations)

  u_boundary = boundary_condition.boundary_value_function(u_inner, orientation_or_normal, equations)

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation_or_normal, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation_or_normal, equations)
  end

  return flux
end

# Wall boundary condition for use with UnstructuredMesh2D
# Note: For unstructured we lose the concept of an "absolute direction"
@inline function (boundary_condition::BoundaryConditionWall)(u_inner,
                                                             normal_direction::AbstractVector,
                                                             x, t,
                                                             surface_flux_function, equations)
  # get the external value of the solution
  u_boundary = boundary_condition.boundary_value_function(u_inner, normal_direction, equations)

  flux = surface_flux_function(u_inner, u_boundary, normal_direction, equations)

  return flux
end

# Specific slip wall boundary condition for use with UnstructuredMesh2D
# Note: For unstructured we lose the concept of an "absolute direction"
@inline function (boundary_condition::BoundaryConditionWall)(u_inner,
                                                             normal_direction::AbstractVector,
                                                             x, t,
                                                             surface_flux_function, equations)
  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal = normal_direction / norm_

  # get the external value of the pressue
  p_boundary = boundary_condition.boundary_value_function(u_inner, normal, equations)

  # For this particular slip wall boundary condition we do not require an evaluation
  # of the surface_flux_function and can directly set the flux because the normal velocity
  # has been set to zero
  return SVector{nvariables(equations)}(0.0, (p_boundary .* normal)..., 0.0) * norm_

end


# FROM compressible_euler_2d.jl

"""
    boundary_state_slip_wall(u_internal, normal_direction::AbstractVector,
                             equations::CompressibleEulerEquations2D)

Determine the external solution value for a slip wall condition. Sets the normal
velocity of the exterior fictitious element to the negative of the internal value.
Density is taken from the internal solution state and pressure is computed as an
exact solution of a 1D Riemann problem. Further details about this boundary state
are available in the paper:
- J. J. W. van der Vegt and H. van der Ven (2002)
  Slip flow boundary conditions in discontinuous Galerkin discretizations of
  the Euler equations of gas dynamics
  [PDF](https://reports.nlr.nl/bitstream/handle/10921/692/TP-2002-300.pdf?sequence=1)

Details about the 1D pressure Riemann solution can be found in Section 6.3.3 of the book
- Eleuterio F. Toro (2009)
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
  3rd edition
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

!!! warning "Experimental code"
    This wall function can change any time.
"""
@inline function boundary_state_slip_wall(u_internal, normal_direction::AbstractVector,
                                          equations::CompressibleEulerEquations2D)

  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # rotate the internal solution state
  u_local = rotate_to_x(u_internal, normal, equations)

  # compute the primitive variables
  rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

  # get the solution of the pressure Riemann problem
  if v_normal <= 0.0
    sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
    p_star = p_local * (1.0 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2.0 * equations.gamma * equations.inv_gamma_minus_one)
  else # v_normal > 0.0
    A = 2.0 / ((equations.gamma + 1) * rho_local)
    B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
    p_star = p_local + 0.5 * v_normal / A * (v_normal + sqrt(v_normal^2 + 4.0 * A * (p_local + B)))
  end

  # compute the conservative variables of the rotated external state
  # Note that the normal velocity component changes sign in the rotated coordinate system
  u_external = prim2cons(SVector(rho_local, -v_normal, v_tangent, p_star), equations)

  # back rotate and return the newly created external state vector
  return rotate_from_x(u_external, normal, equations)
end


# FROM acoustic_perturbation_2d.jl

"""
    boundary_state_slip_wall(u_inner, normal_direction::AbstractVector,
                             equations::AcousticPertubationEquations2D)

Idea behind this boundary condition is to use an orthogonal projection of the perturbed velocities
to zero out the normal velocity while retaining the possibility of a tangential velocity
in the boundary state. Further details are available in the paper:
- Marcus Bauer, Jürgen Dierke and Roland Ewert (2011)
  Application of a discontinuous Galerkin method to discretize acoustic perturbation equations
  [DOI: 10.2514/1.J050333](https://doi.org/10.2514/1.J050333)
"""
function boundary_state_slip_wall(u_inner, normal_direction::AbstractVector,
                                  equations::AcousticPerturbationEquations2D)
  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # compute the normal and tangential components of the velocity
  u_normal  = normal[1] * u_inner[1] + normal[2] * u_inner[2]

  return SVector(u_inner[1] - 2.0 * u_normal * normal[1],
                 u_inner[2] - 2.0 * u_normal * normal[2],
                 u_inner[3],
                 cons2mean(u_inner, equations)...)
end


#FROM compressible_euler_3d.jl

"""
    boundary_state_slip_wall(u_internal, normal_direction::AbstractVector,
                             equations::CompressibleEulerEquations3D)

Determine the external solution value for a slip wall condition. Sets the normal
velocity of the exterior fictitious element to the negative of the internal value.
Density is taken from the internal solution state and pressure is computed as an
exact solution of a 1D Riemann problem. Further details about this boundary state
are available in the paper:
- J. J. W. van der Vegt and H. van der Ven (2002)
  Slip flow boundary conditions in discontinuous Galerkin discretizations of
  the Euler equations of gas dynamics
  [PDF](https://reports.nlr.nl/bitstream/handle/10921/692/TP-2002-300.pdf?sequence=1)

Details about the 1D pressure Riemann solution can be found in Section 6.3.3 of the book
- Eleuterio F. Toro (2009)
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
  3rd edition
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

!!! warning "Experimental code"
    This wall function can change any time.
"""
@inline function boundary_state_slip_wall(u_internal, normal_direction::AbstractVector,
                                          equations::CompressibleEulerEquations3D)

  # normalize the outward pointing direction
  normal = normal_direction / norm(normal_direction)

  # Some vector that can't be identical to normal_vector (unless normal_vector == 0)
  tangent1 = SVector(normal_direction[2], normal_direction[3], -normal_direction[1])
  # Orthogonal projection
  tangent1 -= dot(normal, tangent1) * normal
  tangent1 = normalize(tangent1)

  # Third orthogonal vector
  tangent2 = normalize(cross(normal_direction, tangent1))

  # rotate the internal solution state
  u_local = rotate_to_x(u_internal, normal, tangent1, tangent2, equations)

  # compute the primitive variables
  rho_local, v_normal, v_tangent1, v_tangent2, p_local = cons2prim(u_local, equations)

  # get the solution of the pressure Riemann problem
  if v_normal <= 0.0
    sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
    p_star = p_local * (1.0 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2.0 * equations.gamma * equations.inv_gamma_minus_one)
  else # v_normal > 0.0
    A = 2.0 / ((equations.gamma + 1) * rho_local)
    B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
    p_star = p_local + 0.5 * v_normal / A * (v_normal + sqrt(v_normal^2 + 4.0 * A * (p_local + B)))
  end

  # compute the conservative variables of the rotated external state
  # Note that the normal velocity component changes sign in the rotated coordinate system
  u_external = prim2cons(SVector(rho_local, -v_normal, v_tangent1, v_tangent2, p_star), equations)

  # back rotate and return the newly created external state vector
  return rotate_from_x(u_external, normal, tangent1, tangent2, equations)
end