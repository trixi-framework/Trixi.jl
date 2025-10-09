"""
    reconstruction_constant(u_ll, u_lr, u_rl, u_rr,
                            x_interfaces,
                            node_index, limiter, dg)

Returns the constant "reconstructed" values `u_lr, u_rl` at the interface `x_interfaces[node_index - 1]`.
Supposed to be used in conjunction with [`VolumeIntegralPureLGLFiniteVolumeO2`](@ref).
Formally first order accurate.
If a first-order finite volume scheme is desired, [`VolumeIntegralPureLGLFiniteVolume`](@ref) is an
equivalent, but more efficient choice.
"""
@inline function reconstruction_constant(u_ll, u_lr, u_rl, u_rr,
                                         x_interfaces, node_index,
                                         limiter, dg)
    return u_lr, u_rl
end

# Helper functions for reconstructions below
@inline function reconstruction_linear(u_lr, u_rl, s_l, s_r,
                                       x_lr, x_rl, x_interfaces, node_index)
    # Linear reconstruction at the interface
    u_lr = u_lr + s_l * (x_interfaces[node_index - 1] - x_lr)
    u_rl = u_rl + s_r * (x_interfaces[node_index - 1] - x_rl)

    return u_lr, u_rl
end

#             Reference element:             
#  -1 ------------------0------------------ 1 -> x
# Gauss-Lobatto-Legendre nodes (schematic for k = 3):
#   .          .                  .         .
#   ^          ^                  ^         ^
# Node indices:
#   1          2                  3         4
# The inner subcell boundaries are governed by the
# cumulative sum of the quadrature weights - 1 .
#  -1 ------------------0------------------ 1 -> x
#        w1-1      (w1+w2)-1   (w1+w2+w3)-1
#   |     |             |             |     |
# Note that only the inner boundaries are stored.
# Subcell interface indices, loop only over 2 -> nnodes(dg) = 4
#   1     2             3             4     5
#
# In general a four-point stencil is required, since we reconstruct the
# piecewise linear solution in both subcells next to the subcell interface.
# Since these subcell boundaries are not aligned with the DG nodes,
# on each neighboring subcell two linear solutions are reconstructed => 4 point stencil.
# For the outer interfaces the stencil shrinks since we do not consider values 
# outside the element (volume integral).
# 
# The left subcell node values are labelled `_ll` (left-left) and `_lr` (left-right), while
# the right subcell node values are labelled `_rl` (right-left) and `_rr` (right-right).

"""
    reconstruction_O2_full(u_ll, u_lr, u_rl, u_rr,
                           x_interfaces, node_index,
                           limiter, dg::DGSEM)

Returns the reconstructed values `u_lr, u_rl` at the interface `x_interfaces[node_index - 1]`.
Computes limited (linear) slopes on the subcells for a DGSEM element.
Supposed to be used in conjunction with [`VolumeIntegralPureLGLFiniteVolumeO2`](@ref).

The supplied `limiter` governs the choice of slopes given the nodal values
`u_ll`, `u_lr`, `u_rl`, and `u_rr` at the (Gauss-Lobatto Legendre) nodes.
Total-Variation-Diminishing (TVD) choices for the limiter are
    1) [`minmod`](@ref)
    2) [`monotonized_central`](@ref)
    3) [`superbee`](@ref)
    4) [`vanLeer`](@ref)

The reconstructed slopes are for `reconstruction_O2_full` not limited at the cell boundaries.
Formally second order accurate when used without a limiter, i.e., `limiter = `[`central_slope`](@ref).
This approach corresponds to equation (79) described in
- Rueda-Ramírez, Hennemann, Hindenlang, Winters, & Gassner (2021).
  "An entropy stable nodal discontinuous Galerkin method for the resistive MHD equations.
   Part II: Subcell finite volume shock capturing"
  [JCP: 2021.110580](https://doi.org/10.1016/j.jcp.2021.110580)
"""
@inline function reconstruction_O2_full(u_ll, u_lr, u_rl, u_rr,
                                        x_interfaces, node_index,
                                        limiter, dg::DGSEM)
    @unpack nodes = dg.basis
    x_lr = nodes[node_index - 1]
    x_rl = nodes[node_index]

    # Slope between "middle" nodes
    s_m = (u_rl - u_lr) / (x_rl - x_lr)

    if node_index == 2 # Catch case ll == lr
        s_l = s_m # Use unlimited "central" slope
    else
        x_ll = nodes[node_index - 2]
        # Slope between "left" nodes
        s_lr = (u_lr - u_ll) / (x_lr - x_ll)
        # Select slope between extrapolated (left) and crossing (middle) slope
        s_l = limiter.(s_lr, s_m)
    end

    if node_index == nnodes(dg) # Catch case rl == rr
        s_r = s_m # Use unlimited "central" slope
    else
        x_rr = nodes[node_index + 1]
        # Slope between "right" nodes
        s_rl = (u_rr - u_rl) / (x_rr - x_rl)
        # Select slope between crossing (middle) and extrapolated (right) slope
        s_r = limiter.(s_m, s_rl)
    end

    return reconstruction_linear(u_lr, u_rl, s_l, s_r,
                                 x_lr, x_rl, x_interfaces, node_index)
end

"""
    reconstruction_O2_inner(u_ll, u_lr, u_rl, u_rr,
                            x_interfaces, node_index,
                            limiter, dg::DGSEM)

Returns the reconstructed values `u_lr, u_rl` at the interface `x_interfaces[node_index - 1]`.
Computes limited (linear) slopes on the *inner* subcells for a DGSEM element.
Supposed to be used in conjunction with [`VolumeIntegralPureLGLFiniteVolumeO2`](@ref).

The supplied `limiter` governs the choice of slopes given the nodal values
`u_ll`, `u_lr`, `u_rl`, and `u_rr` at the (Gauss-Lobatto Legendre) nodes.
Total-Variation-Diminishing (TVD) choices for the limiter are
    1) [`minmod`](@ref)
    2) [`monotonized_central`](@ref)
    3) [`superbee`](@ref)
    4) [`vanLeer`](@ref)

For the outer, i.e., boundary subcells, constant values are used, i.e, no reconstruction.
This reduces the order of the scheme below 2.
This approach corresponds to equation (78) described in
- Rueda-Ramírez, Hennemann, Hindenlang, Winters, & Gassner (2021).
  "An entropy stable nodal discontinuous Galerkin method for the resistive MHD equations. 
   Part II: Subcell finite volume shock capturing"
  [JCP: 2021.110580](https://doi.org/10.1016/j.jcp.2021.110580)
"""
@inline function reconstruction_O2_inner(u_ll, u_lr, u_rl, u_rr,
                                         x_interfaces, node_index,
                                         limiter, dg::DGSEM)
    @unpack nodes = dg.basis
    x_lr = nodes[node_index - 1]
    x_rl = nodes[node_index]

    # Slope between "middle" nodes
    s_m = (u_rl - u_lr) / (x_rl - x_lr)

    if node_index == 2 # Catch case ll == lr
        # Do not reconstruct at the boundary
        s_l = zero(s_m)
    else
        x_ll = nodes[node_index - 2]
        # Slope between "left" nodes
        s_lr = (u_lr - u_ll) / (x_lr - x_ll)
        # Select slope between extrapolated (left) and crossing (middle) slope
        s_l = limiter.(s_lr, s_m)
    end

    if node_index == nnodes(dg) # Catch case rl == rr
        # Do not reconstruct at the boundary
        s_r = zero(s_m)
    else
        x_rr = nodes[node_index + 1]
        # Slope between "right" nodes
        s_rl = (u_rr - u_rl) / (x_rr - x_rl)
        # Select slope between crossing (middle) and extrapolated (right) slope
        s_r = limiter.(s_m, s_rl)
    end

    return reconstruction_linear(u_lr, u_rl, s_l, s_r,
                                 x_lr, x_rl, x_interfaces, node_index)
end

"""
    central_slope(sl, sr)

Central, non-TVD reconstruction given left and right slopes `sl` and `sr`.
Gives formally full order of accuracy at the expense of sacrificed nonlinear stability.
Similar in spirit to [`flux_central`](@ref).
"""
@inline function central_slope(sl, sr)
    return 0.5f0 * (sl + sr)
end

"""
    minmod(sl, sr)

Classic minmod limiter function for a TVD reconstruction given left and right slopes `sl` and `sr`.
There are many different ways how the minmod limiter can be implemented.
For reference, see for instance Eq. (6.27) in

- Randall J. LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
@inline function minmod(sl, sr)
    return 0.5f0 * (sign(sl) + sign(sr)) * min(abs(sl), abs(sr))
end

"""
    monotonized_central(sl, sr)

Monotonized central limiter function for a TVD reconstruction given left and right slopes `sl` and `sr`.
There are many different ways how the monotonized central limiter can be implemented.
For reference, see for instance Eq. (6.29) in

- Randall J. LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
@inline function monotonized_central(sl, sr)
    # Use recursive property of minmod function
    return minmod(0.5f0 * (sl + sr), minmod(2 * sl, 2 * sr))
end

"""
    superbee(sl, sr)

Superbee limiter function for a TVD reconstruction given left and right slopes `sl` and `sr`.
There are many different ways how the superbee limiter can be implemented.
For reference, see for instance Eq. (6.28) in

- Randall J. LeVeque (2002)
  Finite Volume Methods for Hyperbolic Problems
  [DOI: 10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253)
"""
@inline function superbee(sl, sr)
    return maxmod(minmod(sl, 2 * sr), minmod(2 * sl, sr))
end

"""
    vanLeer(sl, sr)

Symmetric limiter by van Leer.
See for reference page 70 in 

- Siddhartha Mishra, Ulrik Skre Fjordholm and Rémi Abgrall
  Numerical methods for conservation laws and related equations.
  [Link](https://metaphor.ethz.ch/x/2019/hs/401-4671-00L/literature/mishra_hyperbolic_pdes.pdf)
"""
@inline function vanLeer(sl, sr)
    if abs(sl) + abs(sr) > zero(sl)
        return (abs(sr) * sl + abs(sl) * sr) / (abs(sl) + abs(sr))
    else
        return zero(sl)
    end
end
