# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function analyze_coefficient_IDP(mesh::TreeMesh2D, equations, dg, cache, limiter)
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

function analyze_coefficient_IDP(mesh::StructuredMesh{2}, equations, dg, cache, limiter)
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

function analyze_coefficient_MCL(mesh::TreeMesh2D, equations, dg, cache, limiter)
    @unpack weights = dg.basis
    @unpack alpha, alpha_mean, alpha_pressure,
    alpha_mean_pressure, alpha_entropy, alpha_mean_entropy = limiter.cache.subcell_limiter_coefficients

    n_vars = nvariables(equations)

    alpha_avg = zeros(eltype(alpha),
                      n_vars + limiter.PressurePositivityLimiterKuzmin +
                      limiter.SemiDiscEntropyLimiter)
    alpha_mean_avg = zeros(eltype(alpha),
                           n_vars + limiter.PressurePositivityLimiterKuzmin +
                           limiter.SemiDiscEntropyLimiter)
    total_volume = zero(eltype(alpha))

    for element in eachelement(dg, cache)
        jacobian = inv(cache.elements.inverse_jacobian[element])
        for j in eachnode(dg), i in eachnode(dg)
            for v in eachvariable(equations)
                alpha_avg[v] += jacobian * weights[i] * weights[j] *
                                alpha[v, i, j, element]
                alpha_mean_avg[v] += jacobian * weights[i] * weights[j] *
                                     alpha_mean[v, i, j, element]
            end
            if limiter.PressurePositivityLimiterKuzmin
                alpha_avg[n_vars + 1] += jacobian * weights[i] * weights[j] *
                                         alpha_pressure[i, j, element]
                alpha_mean_avg[n_vars + 1] += jacobian * weights[i] * weights[j] *
                                              alpha_mean_pressure[i, j, element]
            end
            if limiter.SemiDiscEntropyLimiter
                k = n_vars + limiter.PressurePositivityLimiterKuzmin + 1
                alpha_avg[k] += jacobian * weights[i] * weights[j] *
                                alpha_entropy[i, j, element]
                alpha_mean_avg[k] += jacobian * weights[i] * weights[j] *
                                     alpha_mean_entropy[i, j, element]
            end
            total_volume += jacobian * weights[i] * weights[j]
        end
    end

    return alpha_avg ./ total_volume, alpha_mean_avg ./ total_volume
end

function analyze_coefficient_MCL(mesh::StructuredMesh{2}, equations, dg, cache,
                                 limiter)
    @unpack weights = dg.basis
    @unpack alpha, alpha_mean, alpha_pressure,
    alpha_mean_pressure, alpha_entropy, alpha_mean_entropy = limiter.cache.subcell_limiter_coefficients

    n_vars = nvariables(equations)

    alpha_avg = zeros(eltype(alpha),
                      n_vars + limiter.PressurePositivityLimiterKuzmin +
                      limiter.SemiDiscEntropyLimiter)
    alpha_mean_avg = zeros(eltype(alpha),
                           n_vars + limiter.PressurePositivityLimiterKuzmin +
                           limiter.SemiDiscEntropyLimiter)
    total_volume = zero(eltype(alpha))

    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            jacobian = inv(cache.elements.inverse_jacobian[i, j, element])
            for v in eachvariable(equations)
                alpha_avg[v] += jacobian * weights[i] * weights[j] *
                                alpha[v, i, j, element]
                alpha_mean_avg[v] += jacobian * weights[i] * weights[j] *
                                     alpha_mean[v, i, j, element]
            end
            if limiter.PressurePositivityLimiterKuzmin
                alpha_avg[n_vars + 1] += jacobian * weights[i] * weights[j] *
                                         alpha_pressure[i, j, element]
                alpha_mean_avg[n_vars + 1] += jacobian * weights[i] * weights[j] *
                                              alpha_mean_pressure[i, j, element]
            end
            if limiter.SemiDiscEntropyLimiter
                k = n_vars + limiter.PressurePositivityLimiterKuzmin + 1
                alpha_avg[k] += jacobian * weights[i] * weights[j] *
                                alpha_entropy[i, j, element]
                alpha_mean_avg[k] += jacobian * weights[i] * weights[j] *
                                     alpha_mean_entropy[i, j, element]
            end
            total_volume += jacobian * weights[i] * weights[j]
        end
    end

    return alpha_avg ./ total_volume, alpha_mean_avg ./ total_volume
end
end # @muladd
