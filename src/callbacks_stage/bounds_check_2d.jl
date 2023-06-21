# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function check_bounds(u, mesh::AbstractMesh{2}, equations, solver, cache,
                              indicator::IndicatorIDP,
                              time, iter, output_directory, save_errors, interval)
    @unpack local_minmax, positivity, spec_entropy, math_entropy = solver.volume_integral.indicator
    @unpack variable_bounds = indicator.cache.container_shock_capturing
    @unpack idp_bounds_delta = indicator.cache

    save_errors_ = save_errors && (iter % interval == 0)
    counter = 1
    if save_errors_
        open("$output_directory/deviations.txt", "a") do f
            print(f, iter, ", ", time)
        end
    end
    if local_minmax
        for index in indicator.local_minmax_variables_cons
            deviation_min = zero(eltype(u))
            deviation_max = zero(eltype(u))
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                deviation_min = max(deviation_min,
                                    variable_bounds[counter][i, j, element] -
                                    u[index, i, j, element])
                deviation_max = max(deviation_max,
                                    u[index, i, j, element] -
                                    variable_bounds[counter + 1][i, j, element])
            end
            idp_bounds_delta[counter] = max(idp_bounds_delta[counter], deviation_min)
            idp_bounds_delta[counter + 1] = max(idp_bounds_delta[counter + 1],
                                                deviation_max)
            if save_errors_
                deviation_min_ = deviation_min
                deviation_max_ = deviation_max
                open("$output_directory/deviations.txt", "a") do f
                    print(f, ", ", deviation_min_, ", ", deviation_max_)
                end
            end
            counter += 2
        end
    end
    if spec_entropy
        deviation_min = zero(eltype(u))
        for element in eachelement(solver, cache), j in eachnode(solver),
            i in eachnode(solver)

            s = entropy_spec(get_node_vars(u, equations, solver, i, j, element),
                             equations)
            deviation_min = max(deviation_min,
                                variable_bounds[counter][i, j, element] - s)
        end
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], deviation_min)
        if save_errors_
            deviation_min_ = deviation_min
            open("$output_directory/deviations.txt", "a") do f
                print(f, ", ", deviation_min_)
            end
        end
        counter += 1
    end
    if math_entropy
        deviation_max = zero(eltype(u))
        for element in eachelement(solver, cache), j in eachnode(solver),
            i in eachnode(solver)

            s = entropy_math(get_node_vars(u, equations, solver, i, j, element),
                             equations)
            deviation_max = max(deviation_max,
                                s - variable_bounds[counter][i, j, element])
        end
        idp_bounds_delta[counter] = max(idp_bounds_delta[counter], deviation_max)
        if save_errors_
            deviation_max_ = deviation_max
            open("$output_directory/deviations.txt", "a") do f
                print(f, ", ", deviation_max_)
            end
        end
        counter += 1
    end
    if positivity
        for index in indicator.positivity_variables_cons
            if (index in indicator.local_minmax_variables_cons)
                continue
            end
            deviation_min = zero(eltype(u))
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                var = u[index, i, j, element]
                deviation_min = max(deviation_min,
                                    variable_bounds[counter][i, j, element] - var)
            end
            idp_bounds_delta[counter] = max(idp_bounds_delta[counter], deviation_min)
            if save_errors_
                deviation_min_ = deviation_min
                open("$output_directory/deviations.txt", "a") do f
                    print(f, ", ", deviation_min_)
                end
                counter += 1
            end
        end
        for variable in indicator.positivity_variables_nonlinear
            deviation_min = zero(eltype(u))
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                var = variable(get_node_vars(u, equations, solver, i, j, element),
                               equations)
                deviation_min = max(deviation_min,
                                    variable_bounds[counter][i, j, element] - var)
            end
            idp_bounds_delta[counter] = max(idp_bounds_delta[counter], deviation_min)
            if save_errors_
                deviation_min_ = deviation_min
                open("$output_directory/deviations.txt", "a") do f
                    print(f, ", ", deviation_min_)
                end
            end
            counter += 1
        end
    end
    if save_errors_
        open("$output_directory/deviations.txt", "a") do f
            println(f)
        end
    end

    return nothing
end

@inline function check_bounds(u, mesh::AbstractMesh{2}, equations, solver, cache,
                              indicator::IndicatorMCL,
                              time, iter, output_directory, save_errors, interval)
    @unpack var_min, var_max = indicator.cache.container_shock_capturing
    @unpack bar_states1, bar_states2, lambda1, lambda2 = indicator.cache.container_bar_states
    @unpack idp_bounds_delta = solver.volume_integral.indicator.cache
    @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.container_antidiffusive_flux

    n_vars = nvariables(equations)

    deviation_min = zeros(eltype(u), n_vars + indicator.PressurePositivityLimiterKuzmin)
    deviation_max = zeros(eltype(u), n_vars)

    if indicator.DensityLimiter
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                deviation_min[1] = max(deviation_min[1],
                                       var_min[1, i, j, element] - u[1, i, j, element])
                deviation_max[1] = max(deviation_max[1],
                                       u[1, i, j, element] - var_max[1, i, j, element])
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        # Checking the bounds for...
        # - density (rho):
        #   \bar{rho}^{min} <= \bar{rho}^{Lim} <= \bar{rho}^{max}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                deviation_min[1] = max(deviation_min[1],
                                       var_min[1, i, j, element] - rho_limited)
                deviation_max[1] = max(deviation_max[1],
                                       rho_limited - var_max[1, i, j, element])
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                deviation_min[1] = max(deviation_min[1],
                                       var_min[1, i, j, element] - rho_limited)
                deviation_max[1] = max(deviation_max[1],
                                       rho_limited - var_max[1, i, j, element])
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                deviation_min[1] = max(deviation_min[1],
                                       var_min[1, i, j, element] - rho_limited)
                deviation_max[1] = max(deviation_max[1],
                                       rho_limited - var_max[1, i, j, element])
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                deviation_min[1] = max(deviation_min[1],
                                       var_min[1, i, j, element] - rho_limited)
                deviation_max[1] = max(deviation_max[1],
                                       rho_limited - var_max[1, i, j, element])
            end
        end
    end # indicator.DensityLimiter

    if indicator.SequentialLimiter
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                for v in 2:n_vars
                    var_limited = u[v, i, j, element] / u[1, i, j, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] - var_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited - var_max[v, i, j, element])
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure = 0.5 *
                                     (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                     u[1, i, j, element] * u[4, i, j, element]
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                end
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        # Checking the bounds for...
        # - velocities and energy (phi):
        #   \bar{phi}^{min} <= \bar{phi}^{Lim} / \bar{rho}^{Lim} <= \bar{phi}^{max}
        # - pressure (p):
        #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2
        var_limited = zero(eltype(idp_bounds_delta))
        error_pressure = zero(eltype(idp_bounds_delta))
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i, j, element] -
                                  antidiffusive_flux1[v, i, j, element] /
                                  lambda1[i, j, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] -
                                           var_limited / rho_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited / rho_limited -
                                           var_max[v, i, j, element])
                    if indicator.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i + 1, j, element] +
                                  antidiffusive_flux1[v, i + 1, j, element] /
                                  lambda1[i + 1, j, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] -
                                           var_limited / rho_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited / rho_limited -
                                           var_max[v, i, j, element])
                    if indicator.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j, element] -
                                  antidiffusive_flux2[v, i, j, element] /
                                  lambda2[i, j, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] -
                                           var_limited / rho_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited / rho_limited -
                                           var_max[v, i, j, element])
                    if indicator.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j + 1, element] +
                                  antidiffusive_flux2[v, i, j + 1, element] /
                                  lambda2[i, j + 1, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] -
                                           var_limited / rho_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited / rho_limited -
                                           var_max[v, i, j, element])
                    if indicator.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
            end
        end
    elseif indicator.ConservativeLimiter
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                for v in 2:n_vars
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] -
                                           u[v, i, j, element])
                    deviation_max[v] = max(deviation_max[v],
                                           u[v, i, j, element] -
                                           var_max[v, i, j, element])
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure = 0.5 *
                                     (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                     u[1, i, j, element] * u[4, i, j, element]
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                end
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        # Checking the bounds for...
        # - conservative variables (phi):
        #   \bar{rho*phi}^{min} <= \bar{rho*phi}^{Lim} <= \bar{rho*phi}^{max}
        # - pressure (p):
        #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2
        var_limited = zero(eltype(idp_bounds_delta))
        error_pressure = zero(eltype(idp_bounds_delta))
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i, j, element] -
                                  antidiffusive_flux1[v, i, j, element] /
                                  lambda1[i, j, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] - var_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited - var_max[v, i, j, element])
                    if indicator.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i + 1, j, element] +
                                  antidiffusive_flux1[v, i + 1, j, element] /
                                  lambda1[i + 1, j, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] - var_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited - var_max[v, i, j, element])
                    if indicator.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j, element] -
                                  antidiffusive_flux2[v, i, j, element] /
                                  lambda2[i, j, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] - var_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited - var_max[v, i, j, element])
                    if indicator.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j + 1, element] +
                                  antidiffusive_flux2[v, i, j + 1, element] /
                                  lambda2[i, j + 1, element]
                    deviation_min[v] = max(deviation_min[v],
                                           var_min[v, i, j, element] - var_limited)
                    deviation_max[v] = max(deviation_max[v],
                                           var_limited - var_max[v, i, j, element])
                    if indicator.PressurePositivityLimiterKuzmin && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if indicator.PressurePositivityLimiterKuzmin
                    error_pressure -= var_limited * rho_limited
                    deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                    error_pressure)
                    error_pressure = zero(eltype(idp_bounds_delta))
                end
            end
        end
    elseif indicator.PressurePositivityLimiterKuzmin
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                error_pressure = 0.5 * (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                 u[1, i, j, element] * u[4, i, j, element]
                deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                error_pressure)
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        # Checking the bounds for...
        # - pressure (p):
        #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                error_pressure = 0.5 *
                                 (bar_states1[2, i, j, element] -
                                  antidiffusive_flux1[2, i, j, element] /
                                  lambda1[i, j, element])^2 +
                                 0.5 *
                                 (bar_states1[3, i, j, element] -
                                  antidiffusive_flux1[3, i, j, element] /
                                  lambda1[i, j, element])^2 -
                                 (bar_states1[4, i, j, element] -
                                  antidiffusive_flux1[4, i, j, element] /
                                  lambda1[i, j, element]) * rho_limited
                deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                error_pressure)
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                error_pressure = 0.5 *
                                 (bar_states1[2, i + 1, j, element] +
                                  antidiffusive_flux1[2, i + 1, j, element] /
                                  lambda1[i + 1, j, element])^2 +
                                 0.5 *
                                 (bar_states1[3, i + 1, j, element] +
                                  antidiffusive_flux1[3, i + 1, j, element] /
                                  lambda1[i + 1, j, element])^2 -
                                 (bar_states1[4, i + 1, j, element] +
                                  antidiffusive_flux1[4, i + 1, j, element] /
                                  lambda1[i + 1, j, element]) * rho_limited
                deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                error_pressure)
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                error_pressure = 0.5 *
                                 (bar_states2[2, i, j, element] -
                                  antidiffusive_flux2[2, i, j, element] /
                                  lambda2[i, j, element])^2 +
                                 0.5 *
                                 (bar_states2[3, i, j, element] -
                                  antidiffusive_flux2[3, i, j, element] /
                                  lambda2[i, j, element])^2 -
                                 (bar_states2[4, i, j, element] -
                                  antidiffusive_flux2[4, i, j, element] /
                                  lambda2[i, j, element]) * rho_limited
                deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                error_pressure)
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                error_pressure = 0.5 *
                                 (bar_states2[2, i, j + 1, element] +
                                  antidiffusive_flux2[2, i, j + 1, element] /
                                  lambda2[i, j + 1, element])^2 +
                                 0.5 *
                                 (bar_states2[3, i, j + 1, element] +
                                  antidiffusive_flux2[3, i, j + 1, element] /
                                  lambda2[i, j + 1, element])^2 -
                                 (bar_states2[4, i, j + 1, element] +
                                  antidiffusive_flux2[4, i, j + 1, element] /
                                  lambda2[i, j + 1, element]) * rho_limited
                deviation_min[n_vars + 1] = max(deviation_min[n_vars + 1],
                                                error_pressure)
            end
        end
    end # indicator.PressurePositivityLimiterKuzmin

    if indicator.DensityPositivityLimiter
        # New solution u^{n+1}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                deviation_min[1] = max(deviation_min[1], -u[1, i, j, element])
            end
        end

        # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
        beta = indicator.DensityPositivityCorrectionFactor
        # Checking the bounds for...
        # - density (rho):
        #   beta * \bar{rho} <= \bar{rho}^{Lim}
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # -x
                rho_limited = (1 - beta) * bar_states1[1, i, j, element] -
                              antidiffusive_flux1[1, i, j, element] /
                              lambda1[i, j, element]
                deviation_min[1] = max(deviation_min[1], -rho_limited)
                # +x
                rho_limited = (1 - beta) * bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                deviation_min[1] = max(deviation_min[1], -rho_limited)
                # -y
                rho_limited = (1 - beta) * bar_states2[1, i, j, element] -
                              antidiffusive_flux2[1, i, j, element] /
                              lambda2[i, j, element]
                deviation_min[1] = max(deviation_min[1], -rho_limited)
                # +y
                rho_limited = (1 - beta) * bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                deviation_min[1] = max(deviation_min[1], -rho_limited)
            end
        end
    end # indicator.DensityPositivityLimiter

    for v in eachvariable(equations)
        idp_bounds_delta[1, v] = max(idp_bounds_delta[1, v], deviation_min[v])
        idp_bounds_delta[2, v] = max(idp_bounds_delta[2, v], deviation_max[v])
    end
    if indicator.PressurePositivityLimiterKuzmin
        idp_bounds_delta[1, n_vars + 1] = max(idp_bounds_delta[1, n_vars + 1],
                                              deviation_min[n_vars + 1])
    end

    if !save_errors || (iter % interval != 0)
        return nothing
    end
    open("$output_directory/deviations.txt", "a") do f
        print(f, iter, ", ", time)
        for v in eachvariable(equations)
            print(f, ", ", deviation_min[v], ", ", deviation_max[v])
        end
        if indicator.PressurePositivityLimiterKuzmin
            print(f, ", ", deviation_min[n_vars + 1])
        end
        println(f)
    end

    return nothing
end
end # @muladd
