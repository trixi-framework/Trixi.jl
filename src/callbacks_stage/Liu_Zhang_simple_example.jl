using LinearAlgebra

function project_to_admissible_set(Z_old, lower_bound, upper_bound)
    return @. max.(lower_bound, min.(Z_old, upper_bound))
end

N = 5
u_avg = collect(LinRange(0.0, 2.0, N))
u_avg[5] = -0.1 # violate positivity

cell_volumes = rand(N)
cell_volumes .*= 2 / sum(cell_volumes)

lower_bound = 0.0
upper_bound = 2.0

# Pseudo-inverse of A ∈ R^{N×1} (stored as length-N vector of ones)
A = cell_volumes
pseudo_invA = (1.0 / dot(A, A)) .* A

# Initialization: Z^k
X = copy(u_avg)
Y = copy(u_avg)
Z = copy(u_avg)

residual = floatmax(Float64)

epsilon = 1e-12
global_integral = sum(cell_volumes .* u_avg)
num_DY_iter = 0
while residual >= epsilon && num_DY_iter < 500
    # @show residual, epsilon    

    # project the dual variable to the admissible set
    X_half = project_to_admissible_set(Z, lower_bound, upper_bound)

    # update the primal variable
    gamma = 1.0
    grad_h = 2 * cell_volumes .* (X_half .- u_avg)
    @. Y = 2 * X_half - Z - gamma * grad_h

    # enforce the constraint that the sum of the cell averages is equal to the total volume
    X .= Y .+ (global_integral - dot(A, Y)) .* pseudo_invA

    # update the dual variable
    delta_X = X .- X_half
    @. Z = Z .+ delta_X

    # calculate norm(Z_new .- Z_old) 
    residual = norm(delta_X .* sqrt.(cell_volumes))

    num_DY_iter += 1
end

@show num_DY_iter

# final projection to the admissible set returns the solution 
X = project_to_admissible_set(Z, lower_bound, upper_bound)

@show norm((X - u_avg) .* sqrt.(cell_volumes))
@show minimum(X)
@show sum((X - u_avg) .* cell_volumes)
@show X
@show u_avg
