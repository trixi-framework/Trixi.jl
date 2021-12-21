using SparseArrays
using LinearAlgebra
using Plots
using IncompleteLU
using IterativeSolvers
using Trixi
using PrettyTables

# Call all other functions to apply the differential filter to the solution of
# the KHI problem given by Trixi or a chosen test function, afterwards plot all
# outputs and the difference
function main(N::Int64,N_Q,a::Float64,b::Float64,filt::Bool,
    filt_para::Array{Float64,1},input_func::Int64,trixi::Bool,
    runtime_without_plots::Bool)

    # Apply the filter to the solution of the KHI problem given by Trixi.
    # This file and the elixir have to be in the same folder
    if trixi
        trixi_include(joinpath(@__DIR__,"elixir_euler_kelvin_helmholtz_instability_filter.jl"))

        # Change parameters to fit the Trixi simulation
        N = polydeg
        # N_Q² is the number of elements of the CGSEM/DGSEM method
        N_Q = [Int(sqrt(size(semi.cache.elements.node_coordinates, 4)))]
        # Calculate the solution on the intervall [a,b]²
        a = semi.cache.elements.node_coordinates[1]
        b = semi.cache.elements.node_coordinates[end]

        # 100² evenly distributed points for plotting
        xy_plot = collect(a:(b-a)/100:b)

        # Apply the filter to the input and return the (un)filtered solution
        # ready for plotting
        u_filt, u_nofilt = apply_filter(N,N_Q[1],a,b,filt,filt_para,xy_plot,
            input_func,trixi,sol[end])

        # Don't plot the solution in this case
        if runtime_without_plots
            @goto no_plotting
        end

        # Turn conservative variables into primitive ones
        u_filt = cons_2_prim(u_filt, gamma)
        u_nofilt = cons_2_prim(u_nofilt, gamma)

        loop = [1; 2; 3; 4]  # For plotting coordinates 1 to 4
        prim = ["rho";"v1";"v2";"p"]

    else
        # 100² evenly distributed points for plotting
        xy_plot = collect(a:(b-a)/100:b)

        ϵ = [0.0 for n in N_Q]
        EOC = copy(ϵ)

        u_nofilt = input_test_function(xy_plot, input_func)
        u_filt = zeros(length(xy_plot), length(xy_plot), length(N_Q))

        loop = copy(N_Q)

    end

    # If trixi = true: plot all 4 coordinates of the (un-)filtered solutions
    # If trixi = false: plot (un-)filtered test functions for all N_Q
    default(legend = true)
    for n = 1:length(loop)
        if trixi
            u_nofilt_plot = u_nofilt[:, :, n]

        else
            # Apply the filter to the input and return the (un)filtered solution
            # ready for plotting
            u_filt[:, :, n] = apply_filter(N,N_Q[n],a,b,filt,filt_para,xy_plot,
                input_func,trixi,[0.0])

            u_nofilt_plot = copy(u_nofilt)

            # Absolute error and EOC for checking the accuracy of the program
            ϵ[n] = maximum(abs.(u_filt[:,:, n] - u_nofilt_plot))
            EOC[n] = (n > 1 ? log(ϵ[n] / ϵ[n-1]) / log(N_Q[n-1] / N_Q[n]) : 0.0)

        end

        ##################################
        #### PLOT UNFILTERED SOLUTION  #####
        ##################################

        # No need to plot the unfiltered solution for all N_Q, it does not change
        if trixi || (!trixi && n == 1)

            s = plot(xy_plot, xy_plot, u_nofilt_plot, st = :surface)
            plot!(s,camera = (50, 75),
                title = trixi ? "unfiltered $(prim[n]) for t=$(tspan[2])" :
                 "unfiltered, N_Q = $(loop[n])")

            #plot!(s,#=zticks = [0.5, 1.0, 1.5, 2.0, 2.5],
            #    zticks = [0.5, 1.0, 1.5, 2.0, 2.5],=#xtickfontsize = 10,
            #    ytickfontsize = 10,ztickfontsize = 10,xlabel = L"x",
            #    ylabel = L"y",zlabel = L"\rho",guidefontsize = 15,camera = (50, 75)#=,
            #    clims = extrema(u_nofilt_plot)=#,zlims = (0.5,2.5))

            display(s)

            c = plot(xy_plot, xy_plot, u_nofilt_plot, st = :contourf)
            plot!(c,title = trixi ? "unfiltered $(prim[n]) for t=$(tspan[2])" :
             "unfiltered, N_Q = $(loop[n])")

            display(c)

        end

        ##################################
        #### PLOT FILTERED SOLUTION  #####
        ##################################

        sFilt = plot(xy_plot, xy_plot, u_filt[:, :, n], st = :surface)
        if filt
            plot!(sFilt,camera = (50, 75),
                title = trixi ?  "$(prim[n]), germano filt, δ² = $(filt_para[3])" :
                    "germano filt, δ² = $(filt_para[3]), N_Q = $(loop[n])")
        else
            plot!(sFilt,camera = (50, 75),
                title = trixi ?  "$(prim[n]), new filt, α² = $(filt_para[1]), β² = $(filt_para[2])" :
                    "new filt, α² = $(filt_para[1]), β² = $(filt_para[2]), N_Q = $(loop[n])")
        end

        display(sFilt)

        cFilt = plot(xy_plot, xy_plot, u_filt[:, :, n], st = :contourf)
        if filt
            plot!(cFilt,
            title = trixi ?  "$(prim[n]), germano filt, δ² = $(filt_para[3])" :
                "germano filt, δ² = $(filt_para[3]), N_Q = $(loop[n])")
        else
            plot!(cFilt,
            title = trixi ?  "$(prim[n]), new filt, α² = $(filt_para[1]), β² = $(filt_para[2])" :
                "new filt, α² = $(filt_para[1]), β² = $(filt_para[2]), N_Q = $(loop[n])")
        end

        display(cFilt)


        ##################################
        #### PLOT THE DIFFERENCE     #####
        ##################################

        diff = u_nofilt_plot - u_filt[:, :, n]

        sDiff = plot(xy_plot, xy_plot, diff, st = :surface)
        plot!(sDiff,camera = (50, 75),
            title = trixi ? "difference for $(prim[n]), t=$(tspan[2])" : "difference for N_Q = $(N_Q[n])")

        display(sDiff)

        cDiff = plot(xy_plot, xy_plot, diff, st = :contourf)
        plot!(cDiff,
            title = trixi ? "difference for $(prim[n]), t=$(tspan[2])" : "difference for N_Q = $(N_Q[n])")

        display(cDiff)

    end

    @label no_plotting

    ### Display the table of errors when testing the accuracy of the program ###
    ### with the method of manufactured solutions                           ###
    if input_func == 3 && !trixi
        header = ["N_Q" "ϵ" "EOC"]
        data = [N_Q ϵ EOC]

        println()
        print(pretty_table(data, header, alignment = :c))
        println()
    end
end

####################

# Choose test function to easily compare the two filters with each other.
# Function 3 is used for testing the accuracy of the program
function input_test_function(xy::Array{Float64,1}, fun::Int64)

    u = zeros(length(xy), length(xy))

    # Some test functions to test the filters on
    if fun == 1             # Function 1
        for i = 1:length(xy)
            for j = 1:length(xy)
                u[i, j] =
                    cos(4 * pi * xy[i]) +
                    0.5 * cos(20 * pi * xy[i]) +
                    1 * sin(4 * pi * xy[j]) + 0.5 * sin(20 * pi * xy[j])
            end
        end
    elseif fun == 2         # Function 2
        for i = 1:length(xy)
            for j = 1:length(xy)
                u[i, j] =
                    -5 * (xy[i]-0.5)^2 - 5 * (xy[j]-0.5)^2 +
                    0.2 * sin(12 * pi * xy[i]) +
                    0.2 * cos(12 * pi * xy[j])
            end
        end

    # Function 3 for testing the accuracy of the program with the method
    # of manufactured solutions
    elseif fun == 3
        for i = 1:length(xy)
            for j = 1:length(xy)
                u[i, j] = sin(2*pi*xy[i]) + sin(2*pi*xy[j])
            end
        end
    end

    return u

end

####################

# Applies the filter to the solution of the KHI problem calculated by Trixi if
# trixi = true, else applies the filter to a chosen function.
# Uses IterativeSolvers.jl for solving the linear system, the solution of which
# is filtered
function apply_filter(N::Int64,N_Q::Int64,a::Float64,b::Float64,filt::Bool,
    filt_para::Array{Float64,1},xy_plot::Array{Float64,1},input_func::Int64,
    trixi::Bool,solution::Array{Float64,1})

    # Calculate matrices of the linear system and other supportive arrays
    LHS, RHS, xy, w_bary = prepare_filter(N, N_Q, a, b, filt, filt_para)

    # Precondition the LHS via the incompleted LU factorization to speed up
    # the iterative solving process
    LHS_precon = ilu(LHS, τ = 0.1)

    # Filter the solution of the KHI problem given by Trixi
    if trixi
        # Matrix of indices that indicate the ordering Trixi uses to store the elements
        ElementMatrix = calc_element_matrix(N_Q)

        # Seperate cooridnates into columns of the matrix u_nofilt_vec
        u_nofilt_vec = rearrange_entries(solution, ElementMatrix, N, N_Q)
        u_nofilt = zeros(N_Q * (N + 1), N_Q * (N + 1), 4)
        u_nofilt_plot = zeros(length(xy_plot), length(xy_plot), 4)
        u_filt_vec = zero(u_nofilt_vec)
        u_filt = zero(u_nofilt)
        u_filt_plot = zero(u_nofilt_plot)

        # Solve the linear system for each of the 4 coordinates of the Trixi input,
        # turn the filtered/unfiltered vectors into matrices and interpolate both
        # to evenly distributed points for plotting
        for k = 1:4
            u_nofilt[:, :, k] = change_shape(u_nofilt_vec[:, k], N, N_Q,false)
            u_nofilt_plot[:, :, k] = prepare_for_plotting(N, N_Q, xy_plot, xy, w_bary, u_nofilt[:, :, k])

            u_nofilt_vec[:, k] = RHS * u_nofilt_vec[:, k]
            u_filt_vec[:, k] = gmres(LHS, u_nofilt_vec[:, k], Pl = LHS_precon)
            u_filt[:, :, k] = change_shape(u_filt_vec[:, k], N, N_Q,false)
            u_filt_plot[:, :, k] = prepare_for_plotting(N, N_Q, xy_plot, xy, w_bary, u_filt[:, :, k])

        end

        return u_filt_plot, u_nofilt_plot

    ##################

    # Filter the test functions given by input_test_function
    else
        u_nofilt = input_test_function(xy,input_func)
        u_nofilt_vec = change_shape(u_nofilt,N,N_Q,true)

        u_filt = zero(u_nofilt)
        u_filt_vec = zero(u_nofilt_vec)
        u_filt_plot = zeros(length(xy_plot), length(xy_plot))

        # Solve the linear system for the input.
        # Turn the filtered vectors into matrices and interpolate them to evenly
        # distributed points for plotting.
        # Test accuracy of the program with input_func = 3 and the following rhs
        if input_func == 3
            u_nofilt_vec = RHS * u_nofilt_vec * (1 + 4*filt_para[3]*pi^2)
        else
            u_nofilt_vec = RHS * u_nofilt_vec
        end

        u_filt_vec = gmres(LHS, u_nofilt_vec, Pl = LHS_precon)

        u_filt = change_shape(u_filt_vec, N, N_Q,false)
        u_filt_plot = prepare_for_plotting(N, N_Q, xy_plot, xy, w_bary, u_filt)

        return u_filt_plot
    end

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

# Calculate matrices and other arrays for solving the linear system of the filter
function prepare_filter(N::Int64,N_Q::Int64,a::Float64,b::Float64,
    filt::Bool,filt_para::Array{Float64,1})

    # Quadrature nodes and weights, barycentric weights on [-1,1]
    xy_quad, w_quad = LegendreGaussLobattoNodesAndWeights(N,4,4*eps())
    w_bary = barycentric_weights(xy_quad)

    # Vector xy with all the grid points on [a,b], grid is in both directions equal
    xy, Δxy = calc_vector_xy(N, N_Q, a, b, xy_quad)

    # Calculates mass matrix M_global and differentiation matrix D_global that
    # are used in the CG method for solving the filter PDE.
    # Both have dimension N_Q² * (N+1)² x N_Q² * (N+1)².
    # We assume perdiodic boundary conditions for all boundaries
    D_global, M_global = calc_global_matrices(N, N_Q, Δxy, xy_quad, w_quad, w_bary)

    # Determine the matrices for the rhs and lhs of the equations, whose solution
    # gives us the filtered vectors
    if filt # Germano filter: (M_global - δ²D_global)*u_filt = M_global*u
        LHS = M_global - filt_para[3] * D_global
        RHS = M_global
    else # New filter: (M_global + α²δ²D_global)*u_filt = (M_global + β²δ²D_global)*u
        LHS = M_global + filt_para[1] * D_global
        RHS = M_global + filt_para[2] * D_global
    end

    return LHS, RHS, xy, w_bary

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

    return xy, Δxy

end

####################

# Calculate the global mass matrix M_global and differentiation matrix D_global
# for the second order filter PDE in 2 dimensions; periodic boundary conditions
function calc_global_matrices(N::Int64,N_Q::Int64,Δxy::Float64,
    xy_quad::Array{Float64,1},w_quad::Array{Float64,1},w_Bary::Array{Float64,1})

    J = (Δxy)^2 / 4  # Determinant of the transformation

    D = diff_matrix(xy_quad, w_Bary)    # Differentiation matrix for one element
    D_scaled = (w_quad .* D)' * D      # Apply quadrature weights to rows

    # Mass matrix M for one element, boundary conditions are considered
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
# x and y directions, assume a periodic boundary for all cases
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

# Turns a N_Q²*(N+1)² vector into a N_Q*(N+1) x N_Q*(N+1) matrix,
# also works the other way around
function change_shape(x, N::Int64, N_Q::Int64, case::Bool)

    # Turn matrix into vector
    if case

        x_temp = [0.0 for i = 1:(N+1)^2, j = 1:N_Q^2]
        count = 1

        for j = 1:N_Q
            for i = 1:N_Q
                temp = reshape(
                    x[(i-1)*(N+1)+1:i*(N+1), (j-1)*(N+1)+1:j*(N+1)],(N + 1)^2,1)
                x_temp[:, count] = temp
                count = count + 1
            end
        end

        u = reshape(x_temp, N_Q^2 * (N + 1)^2, 1)

        # Turn vector into matrix
    else

        x_temp = reshape(x, (N + 1)^2, N_Q^2)
        u = zeros(N_Q * (N + 1), N_Q * (N + 1))
        count = 1

        for j = 1:N_Q
            for i = 1:N_Q
                u[(i-1)*(N+1)+1:i*(N+1), (j-1)*(N+1)+1:j*(N+1)] =
                    reshape(x_temp[:, count], N + 1, N + 1)
                count = count + 1
            end
        end
    end

    return u

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

# Rearrange entries of the vector u to change the Trixi format to the filter format.
# Trixi uses a different order in which it goes through the N_Q² elements than
# the filter. Also seperates the 4 coordinates of the solution given by Trixi,
# each coordinate is a column of the output
function rearrange_entries(u::Array{Float64,1},ElementMatrix::Array{Int64,2},
                           N::Int64,N_Q::Int64)

    U_filter = zeros(N_Q^2 * (N + 1)^2, 4)

    U = reshape(u, 4, Int(length(u) / 4))
    U = convert(Array{Float64,2}, U')

    ElementList = Int.(vec(reshape(ElementMatrix, N_Q^2, 1)))

    # indexT according to Trixi order, indexF according to the order of the filter
    indexT = 1
    while indexT <= N_Q^2
        sol_matrix =
            reshape(U[1+(indexT-1)*(N+1)^2:indexT*(N+1)^2, :], N + 1, N + 1, 4)
        sol_vec = reshape(permutedims(sol_matrix, (2, 1, 3)), (N + 1)^2, 4)

        # Position according to the Trixi order of elements
        position = findall(x -> x == indexT, ElementList)
        indexF = position[1][1] # New index according to the filter order

        U_filter[1+(indexF-1)*(N+1)^2:indexF*(N+1)^2, :] = sol_vec

        indexT = indexT + 1
    end

    return U_filter
end

####################

# Translates grid points in [-1,1] to points in [a,b]
function extend_grid_points(x_MP::Float64, Δx::Float64, ξ::Array{Float64,1})

    return x_MP .+ Δx / 2 * ξ

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

    N = length(x) - 1
    w = [1.0 for i in x]

    for j = 1:N+1
        for i = 1:N+1
            if i != j
                w[j] = w[j] * (x[j] - x[i])
            end
        end
    end

    return 1 ./ w

end

####################

# Matrix for visualization purposes
function visual_matrix(
    x::Array{Float64,1},
    w::Array{Float64,1},
    z::Array{Float64,1})

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

####################

# Combined algorithm to compute L_N(x), q(x)=L_(N+1)-L_(N-1) and q'(x)
function qAndLEvaluation(N::Int64,x::Float64)

    if N < 2
        println("Error! N must be at least 2")

    else

        L = 0
        L_2 = 1
        L_1 = x
        L_diff_2 = 0
        L_diff_1 = 1

        for k=2:N
            L = (2*k-1)/k * x * L_1 - (k-1)/k * L_2
            L_diff = L_diff_2 + (2*k-1) * L_1

            L_2 = L_1
            L_1 = L
            L_diff_2 = L_diff_1
            L_diff_1 = L_diff

        end

        k = N+1

        L_final = (2*k-1)/k * x * L - (k-1)/k * L_2
        L_diff_final = L_diff_2 + (2*k-1) * L_1

        q = L_final - L_2
        q_diff = L_diff_final - L_diff_2

        return q, q_diff, L

    end

end

####################

function LegendreGaussLobattoNodesAndWeights(N::Int64,n_it::Int64,TOL::Float64)

    if N < 1
        println("Error! N must be at least 1")

    elseif N == 1
        x = [-1.0 1.0]
        w = [1.0 1.0]

    else

        x = [i == 1 ? -1.0 : i == N+1 ? 1.0 : 0.0 for i=1:N+1]
        w = [i == 1 || i == N+1 ? 2/(N*(N+1)) : 0.0 for i=1:N+1]

        for j=1:Int(floor((N+1)/2))-1
            x[j+1] = -cos((j+1/4)*π/N - 3/(8*N*π) * 1/(j+1/4))

            for k=0:n_it
                q, q_diff, L = qAndLEvaluation(N,x[j+1])
                Δ = -q/q_diff
                x[j+1] = x[j+1] + Δ

                abs(Δ) <= TOL * abs(x[j+1]) ? break : nothing

            end

            q, q_diff, L = qAndLEvaluation(N,x[j+1])
            x[N+1-j] = -x[j+1]
            w[j+1] = 2/(N*(N+1)*L^2)
            w[N+1-j] = w[j+1]

        end
    end

    if N > 1 && N % 2 == 0
        q, q_diff, L = qAndLEvaluation(N,0.0)
        x[Int(N/2)+1] = 0.0
        w[Int(N/2)+1] = 2/(N*(N+1)*L^2)

    end

    return x, w

end

