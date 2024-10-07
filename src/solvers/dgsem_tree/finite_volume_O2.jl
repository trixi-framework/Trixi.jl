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

@inline function linear_reconstruction(u_ll, u_rr, s_l, s_r,
                                       x_ll, x_rr, x_interfaces, node_index)
    # Linear reconstruction at the interface
    u_ll = u_ll + s_l * (x_interfaces[node_index - 1] - x_ll)
    u_rr = u_rr + s_r * (x_interfaces[node_index - 1] - x_rr)

    return u_ll, u_rr
end

#             Reference element:             
#  -1 -----------------0----------------- 1 -> x
# Gauss-Lobatto-Legendre nodes (schematic for k = 3):
#   .         .                 .         .
#   ^         ^                 ^         ^
# i - 2,    i - 1,              i,      i + 1
# mm        ll                  rr      pp
# Cell boundaries (schematic for k = 3): 
# (note that only the inner three boundaries are stored)
#  -1 -----------------0----------------- 1 -> x
#   |     |            |             |    |
# Cell index:
#      1         2              3       4

"""
    reconstruction_small_stencil_full(u_mm, u_ll, u_rr, u_pp,
                                      x_interfaces, node_index, limiter, dg)

Computes limited (linear) slopes on the subcells for a DGSEM element.
The supplied `limiter` governs the choice of slopes given the nodal values
`u_mm`, `u_ll`, `u_rr`, and `u_pp` at the (Gauss-Lobatto Legendre) nodes.
Formally O(2) accurate when used without a limiter, i.e., the `central_slope`.
For the subcell boundaries the reconstructed slopes are not limited.
"""
@inline function reconstruction_small_stencil_full(u_mm, u_ll, u_rr, u_pp,
                                                   x_interfaces, node_index,
                                                   limiter, dg)
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
    reconstruction_small_stencil_inner(u_mm, u_ll, u_rr, u_pp,
                                       x_interfaces, node_index, limiter, dg)

Computes limited (linear) slopes on the *inner* subcells for a DGSEM element.
The supplied `limiter` governs the choice of slopes given the nodal values
`u_mm`, `u_ll`, `u_rr`, and `u_pp` at the (Gauss-Lobatto Legendre) nodes.
For the outer, i.e., boundary subcells, constant values are used, i.e, no reconstruction.
This reduces the order of the scheme below 2.
"""
@inline function reconstruction_small_stencil_inner(u_mm, u_ll, u_rr, u_pp,
                                                    x_interfaces, node_index,
                                                    limiter, dg)
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
"""
@inline function central_slope(sl, sr)
    s = 0.5 * (sl + sr)
    return s
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
    return 0.5 * (sign(sl) + sign(sr)) * min(abs(sl), abs(sr))
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
    # CARE: MC assumes equidistant grid in 0.5 * abs(sl + sr)!
    # Use recursive property of minmod function
    s = minmod(0.5 * (sl + sr), minmod(2 * sl, 2 * sr))

    return s
end

# Note: This is NOT a limiter, just a helper for the `superbee` limiter below.
@inline function maxmod(sl, sr)
    return 0.5 * (sign(sl) + sign(sr)) * max(abs(sl), abs(sr))
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
    s = maxmod(minmod(sl, 2 * sr), minmod(2 * sl, sr))
    return s
end

"""
    vanLeer_limiter(sl, sr)

Symmetric limiter by van Leer.
See for reference page 70 in 

- Siddhartha Mishra, Ulrik Skre Fjordholm and Rémi Abgrall
  Numerical methods for conservation laws and related equations.
  [Link](https://metaphor.ethz.ch/x/2019/hs/401-4671-00L/literature/mishra_hyperbolic_pdes.pdf)
"""
@inline function vanLeer_limiter(sl, sr)
    if abs(sl) + abs(sr) > 0.0
        s = (abs(sr) * sl + abs(sl) * sr) / (abs(sl) + abs(sr))
    else
        s = 0.0
    end

    return s
end
