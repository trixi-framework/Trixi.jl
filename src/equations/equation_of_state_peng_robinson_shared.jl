#! format: noindent

const PengRobinsonEOS = Union{PengRobinson, HelmholtzPengRobinson}

# the default tolerance of 10 * eps() does not converge for most Peng-Robinson examples,
# so we choose a looser tolerance here. Researchers at the US Naval Research Lab noted
# that they typically just use 8 fixed Newton iterations for Peng-Robinson.
eos_newton_tol(eos::PengRobinsonEOS) = convert(eltype(eos.R), 1e-8)

@inline function peng_robinson_a(T, eos::PengRobinsonEOS)
    (; a0, kappa, Tc) = eos
    return a0 * (1 + kappa * (1 - sqrt(T / Tc)))^2
end

@inline peng_robinson_da(T, eos::PengRobinsonEOS) = ForwardDiff.derivative(T -> peng_robinson_a(T,
                                                                                                eos),
                                                                           T)

@inline peng_robinson_d2a(T, eos::PengRobinsonEOS) = ForwardDiff.derivative(T -> peng_robinson_da(T,
                                                                                                  eos),
                                                                            T)

@inline function calc_K(V, eos::PengRobinsonEOS)
    (; inv2sqrt2b, one_minus_sqrt2_b, one_plus_sqrt2_b) = eos
    return inv2sqrt2b * log((V + one_minus_sqrt2_b) / (V + one_plus_sqrt2_b))
end
