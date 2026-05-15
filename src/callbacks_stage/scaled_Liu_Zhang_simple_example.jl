using LinearAlgebra

function project_to_admissible_set(Z_old, lower_bound)
    return @. max.(lower_bound, Z_old)
end

N = 2^8
u_avg = rand(N)
u_avg[5] = -0.05 # violate positivity
u_avg[10] = -0.01 # violate positivity
u_avg[20] = -0.1 # violate positivity

function apply_liu_zhang_limiter(u_avg; lower_bound = 0.0)
    N = length(u_avg)

    # cell_volumes = rand(N)
    cell_volumes = ones(N)
    cell_volumes[1:end÷2] *= 0.5
    cell_volumes[1:end÷4] *= 0.5
    cell_volumes[1:end÷8] *= 0.5
    cell_volumes[1:end÷16] *= 0.5
    # cell_volumes[1:end÷32] *= 0.5
    # cell_volumes[1:end÷64] *= 0.5
    # cell_volumes[1:end÷128] *= 0.5
    # cell_volumes[1:end÷256] *= 0.5    
    cell_volumes .*= 2 / sum(cell_volumes)

    # enforce conservation 
    global_integral = sum(cell_volumes .* u_avg)

    sqrt_cell_volumes = sqrt.(cell_volumes)

    # minimize ||X - u_avg_sqrt_volume_weighted||_2
    # and recover u_avg = X / sqrt.(cell_volumes)
    u_avg_sqrt_volume_weighted = u_avg .* sqrt_cell_volumes

    # Initialization: Z^k
    X, Y, Z = ntuple(i -> copy(u_avg_sqrt_volume_weighted), 3)

    residual = floatmax(Float64)

    epsilon = 1e-12
    num_DY_iter = 0
    while residual >= epsilon && num_DY_iter < 100
        # @show residual, epsilon    

        # project the dual variable to the admissible set
        X_half = project_to_admissible_set(Z ./ sqrt_cell_volumes, lower_bound) .* sqrt_cell_volumes

        # # update the primal variable
        # gamma = inv(maximum(cell_volumes))
        # grad_h = 2 * cell_volumes .* (X_half .- u_avg)
        # @. Y = 2 * X_half - Z - gamma * grad_h
        
        @. Y = 2 * X_half - Z - (X_half - u_avg_sqrt_volume_weighted)

        # enforce the constraint that the sum of the cell averages is equal to the total volume
        #X .= Y .+ (global_integral - dot(A, Y)) .* pseudo_invA
        X .= Y .+ sqrt_cell_volumes * (global_integral - dot(sqrt_cell_volumes, Y)) / sum(cell_volumes)

        # update the dual variable
        delta_X = X .- X_half
        @. Z = Z .+ delta_X

        # calculate norm(Z_new .- Z_old) 
        residual = norm(delta_X)

        # check history
        X = project_to_admissible_set(Z ./ sqrt_cell_volumes, lower_bound) .* sqrt_cell_volumes
        rel_cons_err = sum((X - u_avg_sqrt_volume_weighted) .* sqrt_cell_volumes) / global_integral
        @show abs(rel_cons_err), residual

        num_DY_iter += 1
    end

    # final projection to the admissible set returns the solution 
    X = project_to_admissible_set(Z, lower_bound .* sqrt_cell_volumes)

    # @show num_DY_iter
    # @show norm((X - u_avg) .* sqrt.(cell_volumes))
    # @show minimum(X)
    # @show sum((X - u_avg) .* cell_volumes)

    return X, cell_volumes, num_DY_iter
end

X_sqrt_volume_weighted, cell_volumes, num_DY_iter = apply_liu_zhang_limiter(u_avg; lower_bound = 1e-6);
X = X_sqrt_volume_weighted ./ sqrt.(cell_volumes)
sum(@. cell_volumes * (X - u_avg)^2)