#! format: noindent

const VanDerWaalsEOS = Union{VanDerWaals, HelmholtzVanDerWaals}

@doc raw"""
    temperature(V, e_internal, eos::VanDerWaalsEOS)

For van der Waals fluids, ``e_{\text{internal}} = c_v T - a \rho`` with ``\rho = 1/V``,
so ``T = (e_{\text{internal}} + a/V) / c_v``.
"""
function temperature(V, e_internal, eos::VanDerWaalsEOS)
    (; cv, a) = eos
    return (e_internal + a / V) / cv
end
