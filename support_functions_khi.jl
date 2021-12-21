using SparseArrays
using LinearAlgebra
using Plots
using IncompleteLU


######################################################
## FUNCTIONS NECESSARY FOR PLOTTING
#####################################################

# Plot the (un-)filtered solution to the KHI problem calculated by Trixi
function plot_sol(N::Int64,N_Q::Int64,a::Float64,b::Float64,
    xy_quad::Array{Float64,1},w_bary::Array{Float64,1},
    ElementMatrix::Array{Int64,2},solution::Array{Float64,1})


    # Separate coordinates and turn vector into matrix,
    # u is then a matrix of dimension N_Q*(N+1) x N_Q*(N+1) x 4
    u = trixi_vec_to_filter_matrix(solution, ElementMatrix, N, N_Q)
    u = cons_2_prim(u, gamma)

    # Calc grid points xy and evenly spaced points xy_plot for plotting
    xy = calc_vector_xy(N, N_Q, a, b, xy_quad)
    xy_plot = collect(a:(b-a)/100:b)  # Evenly space plotting points

    # Interpolate filtered solution to evenly spaced grid points for plotting
    u_plot = zeros(length(xy_plot), length(xy_plot), 4)
    for k = 1:4
        u_plot[:, :, k] =
            prepare_for_plotting(N, N_Q, xy_plot, xy, w_bary, u[:, :, k])
    end

    # Each column of u_plot represents the values of a coordinate of the solution
    # at the grid points for plotting

    # Plot all 4 coordinates of the (un-)filtered solutions to the KHI problem
    prim = ["rho";"v1";"v2";"p"]
    default(legend = true)
    for k = 1:4

        s = plot(xy_plot, xy_plot, u_plot[:, :, k], st = :surface)
        if filt_type == "step"
            plot!(s,camera = (50, 75),
                title = filt ? "$(prim[k]), step-cb, germano filt, δ² = $(filt_para[3]), T = $(tspan[2])" :
                "$(prim[k]), step-cb, new filt, α² = $(filt_para[1]), β² = $(filt_para[2]), T = $(tspan[2])")
        elseif filt_type == "stage"
            plot!(s,camera = (50, 75),
                title = filt ? "$(prim[k]), stage-cb, germano filt, δ² = $(filt_para[3]), T = $(tspan[2])" :
                "$(prim[k]), stage-cb, new filt, α² = $(filt_para[1]), β² = $(filt_para[2]), T = $(tspan[2])")
        else
            plot!(s,camera = (50, 75),
                title = "$(prim[k]), unfiltered, T = $(tspan[2])")
        end

        display(s)


        c = plot(xy_plot, xy_plot, u_plot[:, :, k], st = :contourf)
        if filt_type == "step"
            plot!(c,title = filt ? "$(prim[k]), step-cb, germano filt, δ² = $(filt_para[3]), T = $(tspan[2])" :
                "$(prim[k]), step-cb, new filt, α² = $(filt_para[1]), β² = $(filt_para[2]), T = $(tspan[2])")
        elseif filt_type == "stage"
            plot!(c,title = filt ? "$(prim[k]), stage-cb, germano filt, δ² = $(filt_para[3]), T = $(tspan[2])" :
                "$(prim[k]), stage-cb, new filt, α² = $(filt_para[1]), β² = $(filt_para[2]), T = $(tspan[2])")
        else
            plot!(c,title = "$(prim[k]), unfiltered, T = $(tspan[2])")
        end

        display(c)

    end

    return nothing

end

####################

# Turn conservative variables of the solution to the KHI problem into primitive ones.
# gamma is needed for calculating the pressure
function cons_2_prim(u,gamma::Float64)

    output = zero(u)

    output[:,:,1] = u[:,:,1]
    output[:,:,2] = u[:,:,2] ./ u[:,:,1]
    output[:,:,3] = u[:,:,3] ./ u[:,:,1]
    output[:,:,4] = (gamma-1) * (u[:,:,4] - 0.5*(u[:,:,2].^2 + u[:,:,3].^2) ./ u[:,:,1])

    return output

end

####################

# Calculate vector of grid points in [a,b] in one dimension
function calc_vector_xy(N::Int64,N_Q::Int64,a::Float64,b::Float64,
    xy_quad::Array{Float64,1})

    Δxy = (b - a) / N_Q
    xy_MP = collect(a+Δxy/2:Δxy:b-Δxy/2)
    xy = [0.0 for i = 1:N_Q*(N+1)]

    # Translate points in [-1,1] to [a,b]
    count = 1
    for i = 1:N+1:length(xy)-N
        xy[i:i+N] = extend_grid_points(xy_MP[count], Δxy, xy_quad)
        count = count + 1
    end

    return xy

end

####################

# Take the solution u in matrix form and interpolate it to evenly distributed
# grid points for plotting
function prepare_for_plotting(N::Int64,N_Q::Int64,xy_plot::Array{Float64,1},
    xy::Array{Float64,1},w_bary::Array{Float64,1},u::Array{Float64,2})

    u_final = zeros(length(xy_plot), N_Q * (N + 1))
    u_output = zeros(length(xy_plot), length(xy_plot))

    # Interpolate in x direction first
    temp = 1
    for i = 1:N+1:length(xy)-N
        if xy[i+N] == xy[end]
            V = visual_matrix(xy[i:end], w_bary, xy_plot[temp:end])
            u_final[temp:end, :] = V * u[i:end, :]

        else

            for j = temp:length(xy_plot)
                if xy_plot[j] <= xy[i+N] && xy_plot[j+1] > xy[i+N]
                    V = visual_matrix(xy[i:i+N], w_bary, xy_plot[temp:j])
                    u_final[temp:j, :] = V * u[i:i+N, :]
                    temp = j + 1
                    break
                end
            end
        end
    end

    u_final = u_final'

    # Interpolate in y direction second
    temp = 1
    for i = 1:N+1:length(xy)-N
        if xy[i+N] == xy[end]
            V = visual_matrix(xy[i:end], w_bary, xy_plot[temp:end])
            u_output[temp:end, :] = V * u_final[i:end, :]

        else

            for j = temp:length(xy_plot)
                if xy_plot[j] <= xy[i+N] && xy_plot[j+1] > xy[i+N]
                    V = visual_matrix(xy[i:i+N], w_bary, xy_plot[temp:j])
                    u_output[temp:j, :] = V * u_final[i:i+N, :]
                    temp = j + 1
                    break
                end
            end
        end
    end

    return u_output'

end

####################

# Take the solution vector calculated by Trixi and separate the coordinates into
# the different columns of U. Then turn each column vector into a matrix
# that can be interpolated to the plotting points
function trixi_vec_to_filter_matrix(u::Array{Float64,1},
    ElementMatrix::Array{Int64,2},N::Int64,N_Q::Int64)

    kron_matrix = spzeros(N_Q, N_Q)
    U_filter = zeros(N_Q * (N + 1), N_Q * (N + 1), 4)

    U = reshape(u, 4, Int(length(u) / 4))
    U = convert(Array{Float64,2}, U')

    count = 1
    while count <= N_Q^2
        sol_matrix =
            reshape(U[1+(count-1)*(N+1)^2:count*(N+1)^2, :], N + 1, N + 1, 4)
        sol_matrix_trans = permutedims(sol_matrix, (2, 1, 3))
        tempIndex = findall(x -> x == count, ElementMatrix)

        kron_matrix[tempIndex[1]] = 1
        for i = 1:4
            U_filter[:, :, i] =
                U_filter[:, :, i] + kron(kron_matrix, sol_matrix_trans[:, :, i])
        end

        kron_matrix[tempIndex[1]] = 0
        kron_matrix = dropzeros(kron_matrix)

        count = count + 1
    end

    return U_filter
end

####################

# Translates grid points in [-1,1] to points in [a,b]
function extend_grid_points(x_MP::Float64, Δx::Float64, ξ::Array{Float64,1})

    return x_MP .+ Δx / 2 * ξ

end

####################

# Matrix for visualization purposes
function visual_matrix(x::Array{Float64,1},w::Array{Float64,1},z::Array{Float64,1})

    # x: Current grid points
    # w: Barycentric weights
    # z: Grid points we want to interpolate to

    N = length(x) - 1
    N_out = length(z) - 1

    V = [0.0 for i = 1:N_out+1, j = 1:N+1]

    for i = 1:N_out+1
        check = false

        for j = 1:N+1
            if z[i] == x[j]
                V[i, j] = 1
                check = true
            end
        end

        if !check
            sum = 0

            for j = 1:N+1
                temp = w[j] / (z[i] - x[j])
                V[i, j] = temp
                sum = sum + temp
            end

            V[i, :] = V[i, :] / sum

        end
    end

    return V
end



######################################################
## FUNCTIONS NECESSARY FOR FILTERING
#####################################################


# Calculate the matrices of the CG method that we need for filtering
function prepareFilter(N::Int64,N_Q::Int64,a::Float64,b::Float64,
    filt::Bool,filt_para::Array{Float64,1},solver::DG)

    # Quadrature nodes and weights, barycentric weights on [-1,1]
    xy_quad = convert(Array{Float64,1},solver.basis.nodes)
    w_quad = convert(Array{Float64,1},solver.basis.weights)
    w_bary = barycentric_weights(xy_quad)

    # Calculates mass matrix M_global and differentiation matrix D_global that
    # are used in the CG method for solving the filter PDE,
    # both have dimension N_Q² * (N+1)² x N_Q² * (N+1)².
    # We assume perdiodic boundary conditions for all boundaries
    D_global, M_global = calc_global_matrices(N, N_Q, a, b, xy_quad, w_quad, w_bary)

    # Rearrange the rows and columns of D_global (M_global is symmetric,
    # no rearranging necessary). The filter uses a different order of elements,
    # so to fit the vector Trixi uses, we habe to change the matrix D_global

    # Matrix of indices that indicate the ordering Trixi uses to store the elements
    ElementMatrix = calc_element_matrix(N_Q)

    # The ordering IN each element is different for Trixi as well, so we have
    # a matrix of indices for that as well
    KnotenMatrix = [0 for i=1:N+1,j=1:N+1]
    for i=1:(N+1)^2
        KnotenMatrix[i] = i
    end
    KnotenMatrix = convert(Array{Int64,2},KnotenMatrix')

    # D_trixi can now be multiplied with vectors given by Trixi
    D_trixi = rearrange_filter_matrix(D_global,ElementMatrix,KnotenMatrix,N,N_Q)

    # Determine the matrices for the rhs and lhs of the equations, whose solution
    # gives us the filtered vectors
    if filt                  # Germano filter: (M - δ²D)*u_filt = M*u
        LHS = M_global - filt_para[3] * D_trixi
        RHS = M_global
    else                     # New filter: (M + α²D)*u_filt = (M + β²D)*u
        LHS = M_global + filt_para[1] * D_trixi
        RHS = M_global + filt_para[2] * D_trixi
    end

    # Factorize the left hand side matrix to speed up the process of LHS\RHS*u,
    # that the callback has to solve over and over again
    LHS_factor = ilu(LHS, τ = 0.1)

    return ElementMatrix, LHS, LHS_factor, RHS, w_bary

end

####################

# Calculate the global matrices for the CG method
function calc_global_matrices(N::Int64,N_Q::Int64,a::Float64,b::Float64,
    xy_quad::Array{Float64,1},w_quad::Array{Float64,1},w_Bary::Array{Float64,1})


    Δxy = (b - a) / N_Q    # Element width
    J = (Δxy)^2 / 4  # Determinant of the transformation

    D = diff_matrix(xy_quad, w_Bary)    # Differentiation matrix for one element
    D_scaled = (w_quad .* D)' * D      # Apply quadrature weights to rows

    # Mass matrix M for one element with boundary conditions accounted for
    M = spdiagm(0 => [(i == 1 || i == N+1 ? w_quad[1]+w_quad[end] : w_quad[i]) for i = 1:N+1])

    # Global mass matrices M_x and M_y for the x and y directions respectively.
    # Periodic boundary conditions, length and width N_Q²*(N+1)²
    M_x = calc_partial_matrices_periodic(N, N_Q, M, D_scaled, 1)
    M_y = calc_partial_matrices_periodic(N, N_Q, M, D_scaled, 2)

    # Global differentiation matrices for the x and y directions.
    # Periodic boundary conditions, length and width N_Q²*(N+1)²
    D_x = calc_partial_matrices_periodic(N, N_Q, M, D_scaled, 3)
    D_y = calc_partial_matrices_periodic(N, N_Q, M, D_scaled, 4)

    # Combine the global matrices for both directions following the 2D CG method
    M_global = J * M_x * M_y
    D_global = -(D_x * M_y + M_x * D_y)

    return D_global, M_global

end

####################

# Calculate 1D mass matrices and differentiation matrices for both the
# x and y directions, assume periodic boundaries for all cases
function calc_partial_matrices_periodic(N::Int64,N_Q::Int64,
    M::SparseMatrixCSC{Float64,Int64},D::Array{Float64,2},case::Int64)

    if case == 1 # Mass matrix for x direction

        H = spdiagm(0 => [1 for i = 1:N_Q*(N+1)])

        Diag_0 = kron(H, M)

        Output = kron(spdiagm(0 => [1 for i = 1:N_Q]), Diag_0)

    elseif case == 2 # Mass matrix for y direction

        H = spdiagm(0 => [1 for i = 1:N_Q^2])
        Diag_0 = kron(M, spdiagm(0 => [1 for i = 1:N+1]))

        Output = kron(H, Diag_0)

    elseif case == 3 # Differentiation matrix for x direction

        Diag_0 = spdiagm(0 => [1 for i = 1:N_Q*(N+1)])

        # Diag_1_in for inner points, Diag_1_out for periodic boundary
        Diag_1_in = spdiagm(N + 1 => [1 for i = 1:(N_Q-1)*(N+1)])
        Diag_1_out = spdiagm(-(N_Q - 1) * (N + 1) => [1 for i = 1:N+1])
        Diag_1 = Diag_1_in + Diag_1_out

        # Diag_min1_in for inner points, Diag_min1_out for periodic boundary
        Diag_min1_in = spdiagm(-(N + 1) => [1 for i = 1:(N_Q-1)*(N+1)])
        Diag_min1_out = spdiagm((N_Q - 1) * (N + 1) => [1 for i = 1:N+1])
        Diag_min1 = Diag_min1_in + Diag_min1_out

        Diag_up = spzeros(N + 1, N + 1)
        Diag_down = copy(Diag_up)
        Diag_up[end, :] = D[1, :]
        Diag_down[1, :] = D[end, :]

        Column =
            kron(Diag_0, D) + kron(Diag_1, Diag_up) + kron(Diag_min1, Diag_down)

        Output = kron(spdiagm(0 => [1 for i = 1:N_Q]), Column)

    elseif case == 4 # Differentiation matrix for x direction

        Diag_0 = spdiagm(0 => [1 for i = 1:N_Q^2])
        Diag_main = kron(D, spdiagm(0 => [1 for i = 1:N+1]))

        # Diag_1_in for inner points, Diag_1_out for periodic boundary
        Diag_1_in = spdiagm(N_Q => [1 for i = 1:(N_Q-1)*N_Q])
        Diag_1_out = spdiagm(-N_Q * (N_Q - 1) => [1 for i = 1:N_Q])
        Diag_1 = Diag_1_in + Diag_1_out

        # Diag_min1_in for inner points, Diag_min1_out for periodic boundary
        Diag_min1_in = spdiagm(-N_Q => [1 for i = 1:(N_Q-1)*N_Q])
        Diag_min1_out = spdiagm(N_Q * (N_Q - 1) => [1 for i = 1:N_Q])
        Diag_min1 = Diag_min1_in + Diag_min1_out

        Diag_up = spzeros((N + 1)^2, (N + 1)^2)
        Diag_down = copy(Diag_up)
        Diag_up[end-N:end, :] = Diag_main[1:N+1, :]
        Diag_down[1:N+1, :] = Diag_main[end-N:end, :]

        Output =
            kron(Diag_0, Diag_main) +
            kron(Diag_1, Diag_up) +
            kron(Diag_min1, Diag_down)

    else
        Output = nothing
        println("Error! Variable case has to be 1, 2, 3 or 4")
    end

    return Output

end

####################

# Rearrange rows and columns of matrix A s.t. it fits the format given by Trixi
function rearrange_filter_matrix(A::SparseMatrixCSC{Float64,Int64},
    ElementMatrix::Array{Int64,2},KnotenMatrix::Array{Int64,2},N::Int64,N_Q::Int64)

    # Rearrange the rows first
    B = zero(A)
    indexF = 1
    while indexF <= N_Q^2

        temp = A[1+(indexF-1)*(N+1)^2:indexF*(N+1)^2, :]

        # Rearrange the in-element order
        for i = 1:N_Q^2
            if length(temp[:, 1+(i-1)*(N+1)^2:i*(N+1)^2].nzval) > 0
                temp2 = temp[:, 1+(i-1)*(N+1)^2:i*(N+1)^2]

                temp3 = zero(temp2)
                for k = 1:(N+1)^2  # In-element rows
                    tempIndex = KnotenMatrix[k]
                    temp3[tempIndex, :] = temp2[k, :]
                end
                temp4 = zero(temp3)
                for k = 1:(N+1)^2  # In-element columns
                    tempIndex = KnotenMatrix[k]
                    temp4[:, tempIndex] = temp3[:, k]
                end
                temp[:, 1+(i-1)*(N+1)^2:i*(N+1)^2] = temp4

            end
        end

        # Find correct position in ElementMatrix
        indexT = ElementMatrix[indexF]

        B[1+(indexT-1)*(N+1)^2:indexT*(N+1)^2, :] = temp

        indexF = indexF + 1

    end

    # Now rearrange the columns
    C = zero(B)
    indexF = 1
    while indexF <= N_Q^2

        temp = B[:, 1+(indexF-1)*(N+1)^2:indexF*(N+1)^2]

        # Find correct position in ElementMatrix
        indexT = ElementMatrix[indexF]

        C[:, 1+(indexT-1)*(N+1)^2:indexT*(N+1)^2] = temp

        indexF = indexF + 1

    end

    return dropzeros(C)
end

####################

# Calculates matrix of indices that indicates the order in which Trixi goes
# through the different elements
function calc_element_matrix(N_Q::Int64)

    hor = [2^(2 * i) for i = 1:Int(log(2, N_Q))-1]
    ver = [2^(2 * i + 1) for i = 1:Int(log(2, N_Q))-1]

    Block = [1 2; 3 4]

    for i = 1:length(ver)
        Block = [Block Block.+hor[i]; Block.+ver[i] Block.+(hor[i]+ver[i])]
    end

    return Block
end

####################

# Calculates the 1D differentiation matrix
function diff_matrix(x::Array{Float64,1}, w_bary::Array{Float64,1})

    N = length(x) - 1

    diag = [0.0 for i in x]  # Entries on the main diagonal

    for i = 1:N+1
        for j = 1:N+1
            if i != j
                diag[i] = diag[i] + w_bary[j] / (x[i] - x[j])
            end
        end

        diag[i] = -1 / w_bary[i] * diag[i]
    end

    D = [i == j ? diag[i] : w_bary[j] / (w_bary[i] * (x[i] - x[j]))
        for i = 1:N+1, j = 1:N+1]

    return D
end

####################

 # Calc barycentric weights for points x
function barycentric_weights(x::Array{Float64,1})

    N = length(x)-1
    w = [1.0 for i in x]

    for j=1:N+1
        for i=1:N+1
            if i != j
                w[j] = w[j] * (x[j] - x[i])
            end
        end
    end

    return 1 ./ w

end

