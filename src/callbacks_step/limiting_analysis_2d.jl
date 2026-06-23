# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function analyze_coefficient(mesh::TreeMesh2D, equations, dg, cache,
                             limiter::SubcellLimiterIDP)
    @unpack weights = dg.basis
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    alpha_avg = zero(eltype(alpha))
    total_volume = zero(eltype(alpha))
    for element in eachelement(dg, cache)
        jacobian = inv(cache.elements.inverse_jacobian[element])
        for j in eachnode(dg), i in eachnode(dg)
            alpha_avg += jacobian * weights[i] * weights[j] * alpha[i, j, element]
            total_volume += jacobian * weights[i] * weights[j]
        end
    end

    return alpha_avg / total_volume
end

function analyze_coefficient(mesh::Union{StructuredMesh{2}, P4estMesh{2}},
                             equations, dg, cache,
                             limiter::SubcellLimiterIDP)
    @unpack weights = dg.basis
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    alpha_avg = zero(eltype(alpha))
    total_volume = zero(eltype(alpha))
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            jacobian = inv(cache.elements.inverse_jacobian[i, j, element])
            alpha_avg += jacobian * weights[i] * weights[j] * alpha[i, j, element]
            total_volume += jacobian * weights[i] * weights[j]
        end
    end

    return alpha_avg / total_volume
end
end # @muladd
