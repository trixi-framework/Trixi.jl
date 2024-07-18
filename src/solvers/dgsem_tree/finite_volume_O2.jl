@inline function linear_reconstruction(u_ll, u_rr, ux_ll, ux_rr, x_ll, x_rr, x_interfaces, node_index)
    # Linear reconstruction at the interface
    u_ll = u_ll + ux_ll * (x_interfaces[node_index - 1] - x_ll)
    u_rr = u_rr + ux_rr * (x_interfaces[node_index - 1] - x_rr)

    return u_ll, u_rr
end

@inline function reconstruction_small_stencil(u_mm, u_ll, u_rr, u_pp,
                                              x_interfaces,
                                              node_index, limiter, dg)
    #             Reference element:             
    #  -1 -----------------0----------------- 1 -> x
    # Gauss Lobatto Legendre nodes (schematic for k = 3):
    #   .         .                 .         .
    #   ^         ^                 ^         ^
    # i - 2,    i - 1,               i,     i + 1
    # mm       ll                   rr     pp
    # Cell boundaries (schematic for k = 3): 
    # (note that only the inner three boundaries are stored)
    #  -1 -----------------0----------------- 1 -> x
    #   |     |            |             |    |
    # Cell index:
    #      1         2              3       4

    @unpack nodes = dg.basis
    x_ll = nodes[node_index - 1]
    x_rr = nodes[node_index]

    # Middle element slope
    ux_m = (u_rr - u_ll) / (x_rr - x_ll)
    if node_index == 2 # Catch case mm == ll
        ux_ll = ux_m
    else
        # Left element slope
        ux_l = (u_ll - u_mm) / (x_ll - nodes[node_index - 2])
        ux_ll = limiter.(ux_l, ux_m)
    end

    if node_index == nnodes(dg) # Catch case rr == pp
        ux_rr = ux_m
    else
        # Right element slope
        ux_r = (u_pp - u_rr) / (nodes[node_index + 1] - x_rr)
        ux_rr = limiter.(ux_m, ux_r)
    end

    linear_reconstruction(u_ll, u_rr, ux_ll, ux_rr, x_ll, x_rr, x_interfaces, node_index)
end

@inline function reconstruction_constant(u_mm, u_ll, u_rr, u_pp,
                                         x_interfaces,
                                         node_index, limiter, dg)
    return u_ll, u_rr
end

@inline function central_recon(sl, sr)
    s = 0.5 * (sl + sr)
    return s
end

@inline function minmod(sl, sr)
    s = 0.0
    if sign(sl) == sign(sr)
        s = sign(sl) * min(abs(sl), abs(sr))
    end
    return s
end

@inline function monotonized_central(sl, sr)
    s = 0.0
    if sign(sl) == sign(sr)
        s = sign(sl) * min(2 * abs(sl), 2 * abs(sr), 0.5 * abs(sl + sr))
    end
    return s
end
