function varnames(::typeof(cons2cons), ::AbstractCompressibleEulerEquations{2})
    ("rho", "rho_v1", "rho_v2", "rho_e")
end
function varnames(::typeof(cons2prim), ::AbstractCompressibleEulerEquations{2})
    ("rho", "v1", "v2", "p")
end

# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, ::AbstractCompressibleEulerEquations{2})
    # cos and sin of the angle between the x-axis and the normalized normal_vector are
    # the normalized vector's x and y coordinates respectively (see unit circle).
    c = normal_vector[1]
    s = normal_vector[2]

    # Apply the 2D rotation matrix with normal and tangent directions of the form
    # [ 1    0    0   0;
    #   0   n_1  n_2  0;
    #   0   t_1  t_2  0;
    #   0    0    0   1 ]
    # where t_1 = -n_2 and t_2 = n_1

    return SVector(u[1],
                   c * u[2] + s * u[3],
                   -s * u[2] + c * u[3],
                   u[4])
end

function varnames(::typeof(cons2cons), ::AbstractCompressibleEulerEquations{3})
    ("rho", "rho_v1", "rho_v2", "rho_v3", "rho_e")
end
function varnames(::typeof(cons2prim), ::AbstractCompressibleEulerEquations{3})
    ("rho", "v1", "v2", "v3", "p")
end

# Rotate normal vector to x-axis; normal, tangent1 and tangent2 need to be orthonormal
# Called inside `FluxRotated` in `numerical_fluxes.jl` so the directions
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, tangent1, tangent2,
                             ::AbstractCompressibleEulerEquations{3})
    # Multiply with [ 1   0        0       0   0;
    #                 0   ―  normal_vector ―   0;
    #                 0   ―    tangent1    ―   0;
    #                 0   ―    tangent2    ―   0;
    #                 0   0        0       0   1 ]
    return SVector(u[1],
                   normal_vector[1] * u[2] + normal_vector[2] * u[3] +
                   normal_vector[3] * u[4],
                   tangent1[1] * u[2] + tangent1[2] * u[3] + tangent1[3] * u[4],
                   tangent2[1] * u[2] + tangent2[2] * u[3] + tangent2[3] * u[4],
                   u[5])
end
