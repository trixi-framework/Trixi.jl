"""
    reconstruction_constant(u_mm, u_ll, u_rr, u_pp,
                            x_interfaces,
                            node_index, limiter, dg)

Returns the constant "reconstructed" values at the interface `x_interfaces[node_index - 1]`
obtained from constant polynomials.
Formally O(1) accurate.
"""
@inline function reconstruction_constant(u_mm, u_ll, u_rr, u_pp,
                                         x_interfaces, node_index,
                                         limiter, dg)
    return u_ll, u_rr
end

# Helper functions for reconstructions below
@inline function linear_reconstruction(u_ll, u_rr, s_l, s_r,
                                       x_ll, x_rr, x_interfaces, node_index)
    # Linear reconstruction at the interface
    u_ll = u_ll + s_l * (x_interfaces[node_index - 1] - x_ll)
    u_rr = u_rr + s_r * (x_interfaces[node_index - 1] - x_rr)

    return u_ll, u_rr
end

#             Reference element:             
#  -1 ------------------0------------------ 1 -> x
# Gauss-Lobatto-Legendre nodes (schematic for k = 3):
#   .          .                  .         .
#   ^          ^                  ^         ^
# i - 2,     i - 1,               i,      i + 1
# mm         ll                   rr      pp
# Cell boundaries (schematic for k = 3) are 
# governed by the cumulative sum of the quadrature weights - 1
# Note that only the inner three boundaries are stored.
#  -1 ------------------0------------------ 1 -> x
#        w1-1      (w1+w2)-1   (w1+w2+w3)-1
#   |     |            |             |      |
# Cell index:
#      1         2             3        4

"""
    reconstruction_O2_full(u_mm, u_ll, u_rr, u_pp,
                           x_interfaces, node_index,
                           limiter, dg::DGSEM)

Computes limited (linear) slopes on the subcells for a DGSEM element.
Supposed to be used in conjunction with [`VolumeIntegralPureLGLFiniteVolumeO2`](@ref).

The supplied `limiter` governs the choice of slopes given the nodal values
`u_mm`, `u_ll`, `u_rr`, and `u_pp` at the (Gauss-Lobatto Legendre) nodes.
Total-Variation-Diminishing (TVD) choices for the limiter are
    1) [`minmod`](@ref)
    2) [`monotonized_central`](@ref)
    3) [`superbee`](@ref)
    4) [`vanLeer`](@ref)

The reconstructed slopes are for `reconstruction_O2_full` not limited at the cell boundaries,
thus overshoots between true mesh elements are possible.
Formally O(2) accurate when used without a limiter, i.e., `limiter = `[`central_slope`](@ref).
"""
@inline function reconstruction_O2_full(u_mm, u_ll, u_rr, u_pp,
                                        x_interfaces, node_index,
                                        limiter, dg::DGSEM)
    @unpack nodes = dg.basis
    x_ll = nodes[node_index - 1]
    x_rr = nodes[node_index]

    # Middle element slope
    s_mm = (u_rr - u_ll) / (x_rr - x_ll)

    if node_index == 2 # Catch case mm == ll
        s_l = s_mm
    else
        # Left element slope
        s_ll = (u_ll - u_mm) / (x_ll - nodes[node_index - 2])
        s_l = limiter.(s_ll, s_mm)
    end

    if node_index == nnodes(dg) # Catch case rr == pp
        s_r = s_mm
    else
        # Right element slope
        s_rr = (u_pp - u_rr) / (nodes[node_index + 1] - x_rr)
        s_r = limiter.(s_mm, s_rr)
    end

    linear_reconstruction(u_ll, u_rr, s_l, s_r, x_ll, x_rr, x_interfaces, node_index)
end

"""
    reconstruction_O2_inner(u_mm, u_ll, u_rr, u_pp,
                            x_interfaces, node_index,
                            limiter, dg::DGSEM)

Computes limited (linear) slopes on the *inner* subcells for a DGSEM element.
Supposed to be used in conjunction with [`VolumeIntegralPureLGLFiniteVolumeO2`](@ref).

The supplied `limiter` governs the choice of slopes given the nodal values
`u_mm`, `u_ll`, `u_rr`, and `u_pp` at the (Gauss-Lobatto Legendre) nodes.
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
@inline function reconstruction_O2_inner(u_mm, u_ll, u_rr, u_pp,
                                         x_interfaces, node_index,
                                         limiter, dg::DGSEM)
    @unpack nodes = dg.basis
    x_ll = nodes[node_index - 1]
    x_rr = nodes[node_index]

    # Middle element slope
    s_mm = (u_rr - u_ll) / (x_rr - x_ll)

    if node_index == 2 # Catch case mm == ll
        # Do not reconstruct at the boundary
        s_l = zero(s_mm)
    else
        # Left element slope
        s_ll = (u_ll - u_mm) / (x_ll - nodes[node_index - 2])
        s_l = limiter.(s_ll, s_mm)
    end

    if node_index == nnodes(dg) # Catch case rr == pp
        # Do not reconstruct at the boundary
        s_r = zero(s_mm)
    else
        # Right element slope
        s_rr = (u_pp - u_rr) / (nodes[node_index + 1] - x_rr)
        s_r = limiter.(s_mm, s_rr)
    end

    linear_reconstruction(u_ll, u_rr, s_l, s_r, x_ll, x_rr, x_interfaces, node_index)
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
    # CARE: MC assumes equidistant grid in 0.5 * (sl + sr)!
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
