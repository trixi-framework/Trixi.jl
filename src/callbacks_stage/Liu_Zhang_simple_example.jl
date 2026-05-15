using LinearAlgebra

lower_bound = 1e-6
upper_bound = 2.0

function project_to_admissible_set(Z_old, lower_bound, upper_bound)
    return @. max.(lower_bound, min.(Z_old, upper_bound))
end

# u_avg = [1.186504953362507
#          1.450165804986605
#          1.4501256333661527
#          1.1864079704915869
#          0.8134950466374934
#          0.549834195013395
#          0.5498743666338478
#          0.8135920295084131]


N = 2^8
u_avg = rand(N)
u_avg[5] = -0.05 # violate positivity
u_avg[10] = -0.01 # violate positivity

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

(u_avg, cell_volumes, global_limiter_tol) = (SVector{1, Float64}[[1.0e-6], [1.0e-6], [1.0e-6], [1.0e-6], [1.0e-6], [1.0e-6], [1.0e-6], [1.0e-6], [0.10249825368945617], [0.3772481870923525], [0.9437444416895546], [0.983778083768607], [1.0064871610902426], [0.9921223885864064], [1.0088317789057237], [0.994897976015139], [0.998688260243438], [0.9989206000897481], [0.9989212057460333], [0.9989212057460333], [0.9989212057460333], [0.9989212057460333], [0.9989212057460329], [0.9989212057460332], [1.027846798067034], [0.49208207774205853], [0.04907185881687067], [0.02915063925229819], [-0.0010234385173525677], [0.0021254451399576367], [-5.993238128652211e-6], [2.470966712772406e-7]], [0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625], 2.220446049250313e-13)
u_avg = getindex.(u_avg, 1)

function apply_liu_zhang_limiter(u_avg, cell_volumes, X=copy(u_avg); lower_bound = 0.0)
    N = length(u_avg)

    # Pseudo-inverse of A ∈ R^{N×1} (stored as length-N vector of ones)
    A = cell_volumes
    pseudo_invA = (1.0 / dot(A, A)) .* A

    # Initialization: Z^k
    # X = copy(u_avg)
    Y = copy(u_avg)
    Z = copy(u_avg)

    residual = floatmax(Float64)

    epsilon = 1e-12
    global_integral = sum(cell_volumes .* u_avg)
    num_DY_iter = 0
    while residual >= epsilon && num_DY_iter < 1000
        # @show residual, epsilon    

        # project the dual variable to the admissible set
        X_half = project_to_admissible_set(Z, lower_bound, upper_bound)

        # update the primal variable
        # gamma = 1.0
        gamma = inv(maximum(cell_volumes))
        grad_h = 2 * cell_volumes .* (X_half .- u_avg)
        @. Y = 2 * X_half - Z - gamma * grad_h

        # enforce the constraint that the sum of the cell averages is equal to the total volume
        X .= Y .+ (global_integral - dot(A, Y)) .* pseudo_invA

        # update the dual variable
        delta_X = X .- X_half
        @. Z = Z .+ delta_X

        # calculate norm(Z_new .- Z_old) 
        residual = norm(delta_X .* sqrt.(cell_volumes))

        @show residual

        num_DY_iter += 1
    end

    # final projection to the admissible set returns the solution 
    X = project_to_admissible_set(Z, lower_bound, upper_bound)

    # @show num_DY_iter
    # @show norm((X - u_avg) .* sqrt.(cell_volumes))
    # @show minimum(X)
    # @show sum((X - u_avg) .* cell_volumes)

    return X, cell_volumes, num_DY_iter
end

X, cell_volumes, num_DY_iter = apply_liu_zhang_limiter(u_avg, cell_volumes; lower_bound = 1e-6);
@show num_DY_iter
sum(@. cell_volumes * (X - u_avg)^2)

# max_num_DY_iter = 0
# for i in 1:1000
#     _, num_DY_iter = apply_liu_zhang_limiter(u_avg; lower_bound = eps())
#     max_num_DY_iter = max(max_num_DY_iter, num_DY_iter)
# end
# @show max_num_DY_iter