using Random: seed!
using OrdinaryDiffEq
using Trixi
using SparseArrays
using LinearAlgebra
using Plots

function testFilter()
    @time trixi_include(get_examples()[29])
end

# define new structs inside a module to allow re-evaluating the file
module TrixiExtensionFilter

using Trixi
using OrdinaryDiffEq
using SparseArrays

# This is an example implementation for a simple stage callback (i.e., a callable
# that is executed after each Runge-Kutta *stage*), which uses the differential
# filter introduced by Germano to filter the solution in every stage
struct FilterStageCallback
    N::Int64
    N_Q::Int64
    LHS
    RHS::SparseMatrixCSC{Float64,Int64}
    #FilterMatrix::Array{Float64,2}

    # You can optionally define an inner constructor like the one below to set up
    # some required stuff. You can also create outer constructors (not demonstrated
    # here) for further customization options.
    function FilterStageCallback(
        N::Int64,
        N_Q::Int64,
        LHS,
        RHS::SparseMatrixCSC{Float64,Int64})
        #FilterMatrix::Array{Float64,2})

        new(N, N_Q, LHS, RHS)
        #new(N, N_Q, FilterMatrix)
    end
end

# This method is called when the `FilterStageCallback` is used as `stage_limiter!`
# which gets called after every RK stage. There is no specific initialization
# method for such `stage_limiter!`s in OrdinaryDiffEq.jl.
function (filter_stage_callback::FilterStageCallback)(u_ode, f, semi, t)

    u = Trixi.wrap_array(u_ode, semi)

    stage_callback_filter(
        u,
        filter_stage_callback.N,
        filter_stage_callback.N_Q,
        filter_stage_callback.LHS,
        filter_stage_callback.RHS,
        #filter_stage_callback.FilterMatrix,
        Trixi.mesh_equations_solver_cache(semi)...)

    return nothing
end

# takes the solution u and applies the differential filter
function stage_callback_filter(
    u,
    N::Int64,
    N_Q::Int64,
    LHS,
    RHS::SparseMatrixCSC{Float64,Int64},
    #FilterMatrix::Array{Float64,2},
    mesh::TreeMesh{2},
    equations,
    dg::DGSEM,
    cache)


    # split u into 4 vectors (the matrix rows) to seperate the coordinates
    # U[k,:] describes the solution at all points for coordinate k
    U = reshape(u, 4, Int(length(u) / 4))
    U = convert(Array{Float64,2},U') # transpose for matrix multiplication

    # apply the filter by solving the following system of equations.
    # the system gets solved for every stage, which takes a lot of time.
    # alternatively you can solve LHS\RHS = FilterMatrix beforehand and just
    # calculate FilterMatrix * U in every stage, but that method takes far longer.
    # LHS is already factorized for better runtime, RHS is sparse
    U_filter = LHS \ (RHS * U)
    #U_filter = FilterMatrix * U

    # the 4 coordinates of the filtered solution, which were seperated, are now
    # being merged into one matrix again
    U_filter = convert(Array{Float64,2},U_filter')
    u_filter = reshape(U_filter,4,N+1,N+1,N_Q^2)

    # overwrite the value of u with the filtered value of u_filter for each
    # node in each element
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = Trixi.get_node_vars(u_filter, equations, dg, i, j, element)
            Trixi.set_node_vars!(u, u_node, equations, dg, i, j, element)
        end
    end

    return nothing

end

end # module TrixiExtensionFilter

import .TrixiExtensionFilter

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

seed!(0)
initial_condition = initial_condition_khi

surface_flux = flux_lax_friedrichs
#volume_flux  = flux_chandrashekar
volume_flux  = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.002,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
# filtering should replace the shock-capturing
#volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                 volume_flux_dg=volume_flux,
#                                                 volume_flux_fv=surface_flux)
volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, -0.5)
coordinates_max = ( 0.5,  0.5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=20,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

stepsize_callback = StepsizeCallback(cfl=1.3)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

# In OrdinaryDiffEq.jl, the `step_limiter!` is called after every Runge-Kutta step
# but before possible RHS evaluations of the new value occur. Hence, it's possible
# to modify the new solution value there without impacting the performance of FSAL
# methods.
# The `stage_limiter!` is called additionally after computing a Runge-Kutta stage
# value but before evaluating the corresponding stage derivatives.
# Hence, if a limiter should be called before each RHS evaluation, it needs to be
# set both as `stage_limiter!` and as `step_limiter!`.
N = polydeg
# N_Q elements in each direction, all in all we have N_Q² elements
N_Q = Int(sqrt(size(semi.cache.elements.node_coordinates,4)))
filt = true  # true for Germano, false for the new filter

# contains the filter parameters α², β² and δ² for both filters
# germano filter: u_f - δ² * u_f" = u
# new filter: u_f + α² * u_f" = u + β² * u"
filterParam = [-0.000003;-0.0000001;5*10^-8]

# calculates the necessary matrices and executes the callback
stage_limiter!, step_limiter!, w_bary, E_Matrix = applyFilter(N, N_Q,
                                                            coordinates_min[1],
                                                            coordinates_max[1],
                                                            filt,
                                                            filterParam,
                                                            solver)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, step_limiter!, williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

# plots the first coordinate (see last argument) of sol
plot_sol(N, N_Q, coordinates_min[1], coordinates_max[1], convert(Array{Float64,1},solver.basis.nodes), w_bary, E_Matrix, sol[end],1)


######################################################
######################################################
########### FUNCTIONS FOR FILTERING #######################
########################################################
#########################################################

# calculates the necessary matrices and executes the callback
function applyFilter(
    N::Int64,
    N_Q::Int64,
    a::Float64,
    b::Float64,
    filt::Bool,
    filterParam::Array{Float64,1},
    solver::DG)

    # calculate necessary matrices/vectors for filtering
    E_Matrix, LHS, RHS, w_bary = prepareFilter(N,N_Q,a,b,filt,filterParam,solver)
    #FilterMatrix, w_bary = prepareFilter(N,N_Q,a,b,filt,filterParam,solver)

    filter_stage_callback! = TrixiExtensionFilter.FilterStageCallback(N,N_Q,LHS,RHS)
    #filter_stage_callback! = TrixiExtensionFilter.FilterStageCallback(N,N_Q,FilterMatrix)

    return filter_stage_callback!, filter_stage_callback!, w_bary, E_Matrix

end

#################################

# calculate the matrices of the CG method that we need for filtering
function prepareFilter(
    N::Int64,
    N_Q::Int64,
    a::Float64,
    b::Float64,
    filt::Bool,
    filterParam::Array{Float64,1},
    solver::DG)

    # quadrature nodes and weights, barycentric weights on [-1,1]
    xy_quad = convert(Array{Float64,1},solver.basis.nodes)
    w_quad = convert(Array{Float64,1},solver.basis.weights)
    w_bary = barycentric_weights(xy_quad)

    # calculates mass matrix M and differentiation matrix D that are used in
    # the CG method for solving the filter PDE
    # both have dimension N_Q² * (N+1)² x N_Q² * (N+1)²
    # we assume perdiodic boundary conditions for all boundaries
    D, M = calc_global_matrices_CG(N, N_Q, a, b, xy_quad, w_quad, w_bary)

    # rearrange the rows and columns of D (M is symmetric, no rearranging necessary)
    # The filter uses a different order of elements, so to fit the vector trixi uses
    # we habe to change the matrix D

    # matrix of indices that indicate the ordering trixi uses to store the elements
    E_Matrix = calc_element_matrix(N_Q)

    # the ordering IN each element is different for trixi as well, so we have
    # a matrix of indeices for that as well
    indexMatrix = Array{Int64,2}(undef,N+1,N+1)
    for i=1:(N+1)^2
        indexMatrix[i] = i
    end
    indexMatrix = convert(Array{Int64,2},indexMatrix')

    # D_trixi can now be multiplied with vectors given by trixi
    D_trixi = rearrange_filter_matrix(D,E_Matrix,indexMatrix,N,N_Q)

    # Determine the matrices for the rhs and lhs of the equations, whose solution
    # gives us the filtered vecors
    if filt # germano filter: (M - δ²D)*u_filt = M*u
        LHS = M - filterParam[3] * D_trixi
        RHS = M
    else # new filter: (M + α²D)*u_filt = (M + β²D)*u
        LHS = M + filterParam[1] * D_trixi
        RHS = M + filterParam[2] * D_trixi
    end

    # factorize the left hand side matrix to speed up the process of LHS\RHS*u,
    # that the callback has to solve over and over again
    #LHS_factor = factorize(LHS)
    LHS_factor = lu(LHS)

    return E_Matrix, LHS_factor, RHS, w_bary
    #return E_Matrix, FilterMatrix, w_bary
end

############################

# calculate the global matrices for the CG method
function calc_global_matrices_CG(
    N::Int64,
    N_Q::Int64,
    a::Float64,
    b::Float64,
    xy_quad::Array{Float64,1},
    w_quad::Array{Float64,1},
    w_Bary::Array{Float64,1})


    Δxy = (b - a) / N_Q    # element width
    J = (Δxy)^2 / 4  # Determinant of the transformation

    D = diffMatrix(xy_quad, w_Bary)    # Differentiation matrix for one element
    D_scaled = (w_quad .* D)' * D      # apply quadrature weights to rows

    # mass matrix M for one element
    M = spdiagm(0 => [(i == 1 || i == N+1 ? w_quad[1]+w_quad[end] : w_quad[i]) for i = 1:N+1])

    # global mass matrices M_x and M_y for the x and y directions respectively
    # periodic boundary conditions, length and width N_Q²*(N+1)²
    M_x = calc_partial_matrices_periodic(N, N_Q, M, D_scaled, 1)
    M_y = calc_partial_matrices_periodic(N, N_Q, M, D_scaled, 2)

    # global differentiation matrices for the x and y directions
    # periodic boundary conditions, length and width N_Q²*(N+1)²
    D_x = calc_partial_matrices_periodic(N, N_Q, M, D_scaled, 3)
    D_y = calc_partial_matrices_periodic(N, N_Q, M, D_scaled, 4)

    # combine the global matrices for both directions following the 2D CG method
    M_global = J * M_x * M_y
    D_global = -(D_x * M_y + M_x * D_y)

    return D_global, M_global

end

#############################################

function calc_partial_matrices_periodic(
    N::Int64,
    N_Q::Int64,
    M::SparseMatrixCSC{Float64,Int64},
    D::Array{Float64,2},
    case::Int64)

    if case == 1 # M_x

        H = spdiagm(0 => [1 for i = 1:N_Q*(N+1)])
        Hauptdiag_M = kron(H, M)

        output = kron(spdiagm(0 => [1 for i = 1:N_Q]), Hauptdiag_M)

    elseif case == 2 # M_y

        H = spdiagm(0 => [1 for i = 1:N_Q^2])
        M_H = kron(M, spdiagm(0 => [1 for i = 1:N+1]))

        output = kron(H, M_H)

    elseif case == 3 # D_x

        Hauptdiag = spdiagm(0 => [1 for i = 1:N_Q*(N+1)])

        # O1 für innere Punkte, O2 für periodischen Rand
        Nebendiag_O1 = spdiagm(N + 1 => [1 for i = 1:(N_Q-1)*(N+1)])
        Nebendiag_O2 = spdiagm(-(N_Q - 1) * (N + 1) => [1 for i = 1:N+1])
        Nebendiag_O = Nebendiag_O1 + Nebendiag_O2

        # U1 für innere Punkte, U2 für periodischen Rand
        Nebendiag_U1 = spdiagm(-(N + 1) => [1 for i = 1:(N_Q-1)*(N+1)])
        Nebendiag_U2 = spdiagm((N_Q - 1) * (N + 1) => [1 for i = 1:N+1])
        Nebendiag_U = Nebendiag_U1 + Nebendiag_U2

        ND_oben = spzeros(N + 1, N + 1)
        ND_unten = copy(ND_oben)
        ND_oben[end, :] = D[1, :]
        ND_unten[1, :] = D[end, :]

        spalteD =
            kron(Hauptdiag, D) +
            kron(Nebendiag_O, ND_oben) +
            kron(Nebendiag_U, ND_unten)

        output = kron(spdiagm(0 => [1 for i = 1:N_Q]), spalteD)

    elseif case == 4 # D_y

        Hauptdiag = spdiagm(0 => [1 for i = 1:N_Q^2])

        # O1 für innere Punkte, O2 für periodischen Rand
        Nebendiag_O1 = spdiagm(N_Q => [1 for i = 1:(N_Q-1)*N_Q])
        Nebendiag_O2 = spdiagm(-N_Q * (N_Q - 1) => [1 for i = 1:N_Q])
        Nebendiag_O = Nebendiag_O1 + Nebendiag_O2

        # U1 für innere Punkte, U2 für periodischen Rand
        Nebendiag_U1 = spdiagm(-N_Q => [1 for i = 1:(N_Q-1)*N_Q])
        Nebendiag_U2 = spdiagm(N_Q * (N_Q - 1) => [1 for i = 1:N_Q])
        Nebendiag_U = Nebendiag_U1 + Nebendiag_U2

        Haupt_D = kron(D, spdiagm(0 => [1 for i = 1:N+1]))

        ND_oben = spzeros((N + 1)^2, (N + 1)^2)
        ND_unten = copy(ND_oben)
        ND_oben[end-N:end, :] = Haupt_D[1:N+1, :]
        ND_unten[1:N+1, :] = Haupt_D[end-N:end, :]

        output =
            kron(Hauptdiag, Haupt_D) +
            kron(Nebendiag_O, ND_oben) +
            kron(Nebendiag_U, ND_unten)

    else
        output = nothing
        println("Fehler! Variable case muss 1, 2, 3 oder 4 sein")
    end

    return output

end

##########################################

# rearrange rows and columns of A s.t. it fits the foramt given by Trixi
function rearrange_filter_matrix(
    A::SparseMatrixCSC{Float64,Int64},
    E_Matrix::Array{Int64,2},
    indexMatrix::Array{Int64,2},
    N::Int64,
    N_Q::Int64)

    # rearrange the rows first
    B = zero(A)
    indexF = 1
    while indexF <= N_Q^2

        temp = A[1+(indexF-1)*(N+1)^2:indexF*(N+1)^2, :]

        # rearrange the in-element order
        for i=1:N_Q^2
            if length(temp[:,1+(i-1)*(N+1)^2:i*(N+1)^2].nzval) > 0
                temp2 = temp[:,1+(i-1)*(N+1)^2:i*(N+1)^2]

                temp3 = zero(temp2)
                for k=1:(N+1)^2  # in-element rows
                    tempIndex = indexMatrix[k]
                    temp3[tempIndex,:] = temp2[k,:]
                end
                temp4 = zero(temp3)
                for k=1:(N+1)^2  # in-element columns
                    tempIndex = indexMatrix[k]
                    temp4[:,tempIndex] = temp3[:,k]
                end
                temp[:,1+(i-1)*(N+1)^2:i*(N+1)^2] = temp4

            end
        end

        # find correct position in E_Matrix
        indexT = E_Matrix[indexF]

        B[1+(indexT-1)*(N+1)^2:indexT*(N+1)^2, :] = temp

        indexF = indexF + 1

    end

    # now rearrange the columns
    C = zero(B)
    indexF = 1
    while indexF <= N_Q^2

        temp = B[:,1+(indexF-1)*(N+1)^2:indexF*(N+1)^2]

        # find correct position in E_Matrix
        indexT = E_Matrix[indexF]

        C[:,1+(indexT-1)*(N+1)^2:indexT*(N+1)^2] = temp

        indexF = indexF + 1

    end

    return dropzeros(C)
end

########################################

# calculates matrix of indices that indicates the order in which trixi goes
# through the different elements
function calc_element_matrix(N_Q::Int64)

    hor = [2^(2*i) for i=1:Int(log(2,N_Q))-1]
    ver = [2^(2*i+1) for i=1:Int(log(2,N_Q))-1]

    Block = [1 2;3 4]

    for i=1:length(ver)
        Block = [Block Block .+ hor[i];Block .+ ver[i] Block .+ (hor[i]+ver[i])]
    end

    return Block
    #return Int.(vec(reshape(Block, N_Q^2, 1)))
end

#######################################

# calculates differentiation matrix D for [-1,1]
function diffMatrix(x::Array{Float64,1},w::Array{Float64,1})

    # w die baryzentrischen Gewichte
    # x die jeweiligen Quadraturstellen

    N = length(x)-1
    diag = [0.0 for i in x]  # Diagonaleinträge von D
    for i=1:N+1
        for j=1:N+1
            if i != j
                diag[i] = diag[i] + w[j]/(x[i]-x[j])
            end
        end
        diag[i] = -1/w[i] * diag[i]
    end
    D = [i == j ? diag[i] : w[j]/(w[i]*(x[i]-x[j])) for i=1:N+1,j=1:N+1]
end

#####################################

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




######################################################
#######################################################
###### FUNCTIONS FOR PLOTTING ###########################
#######################################################
######################################################


function plot_sol(
    N::Int64,
    N_Q::Int64,
    a::Float64,
    b::Float64,
    xy_quad::Array{Float64,1},
    w_bary::Array{Float64,1},
    E_Matrix::Array{Int64,2},
    solution::Array{Float64,1},
    coordinate::Int64)


    # Wandle Lösungsvektor um in Matrixform des Filters
    # u ist hier Matrix der Dimension N_Q*(N+1) x N_Q*(N+1) x 4
    u = VektorTrixiToMatrixFilter(solution, E_Matrix, N, N_Q)

    # bestimme Auswertungsstellen in Filter-Form
    xy = bestimmeVektorX(N, N_Q, a, b, xy_quad, w_bary)
    xy_plot = collect(a:(b-a)/100:b)  # Auswertungsstellen äquidistant

    # interpoliere Lösung auf gleichverteilte Stellen
    u_plot = zeros(length(xy_plot), length(xy_plot), 4)
    for k = 1:4
        u_plot[:, :, k] =
            visualisiereLösung(N, N_Q, xy_plot, xy, w_bary, u[:, :, k])
    end
    u_plot = permutedims(u_plot, (2, 1, 3))

    # die 4 Spalten von u_plot sind die 4 Koordinaten der Lösung ausgewertet
    # an äquidistanten Punkten, diese können nun geplottet werden

    # Plotte ungefilterte Lösung bzw. exakte Lösung
    default(legend = true)

    # Plotte Approximationen verschiedener Genauigkeit
    sFilt = plot(xy_plot, xy_plot, u_plot[:, :, coordinate], st = :surface)
    #plot!(sFilt, camera = (50, 75),
    #    clims = extrema(u_plot[:,:,coordinate]),
    #    zlims = extrema(u_plot[:,:,coordinate]),
    #    title = "T = $(tspan[2]), δ² = $(filterParam[3])")
    plot!(sFilt, camera = (50, 75), clims = (0.75, 2.5), zlims = (0.75, 2.5))
    #savefig("germano_noshock_710_8_S3")

    cFilt = plot(xy_plot, xy_plot, u_plot[:, :, coordinate], st = :contourf)
    #plot!(cFilt,
    #    clims = extrema(u_plot[:,:,coordinate]),
    #    zlims = extrema(u_plot[:,:,coordinate]),
    #    title = "T = $(tspan[2]), δ² = $(filterParam[3])")
    plot!(cFilt, clims = (0.75, 2.5), zlims = (0.75, 2.5))
    #savefig("germano_noshock_710_8_C3")

    display(sFilt)
    display(cFilt)

    return nothing

end

####################

function bestimmeVektorX(
    N::Int64,
    N_Q::Int64,
    a::Float64,
    b::Float64,
    xy_quad::Array{Float64,1},
    w_bary::Array{Float64,1})

    Δxy = (b - a) / N_Q # Durchmesser der Zellen für I = [0,1]^2
    xy_MP = collect(a+Δxy/2:Δxy:b-Δxy/2)  # Mittelpunkte der Zellen zwischen 0 und 1
    xy = [0.0 for i = 1:N_Q*(N+1)] # Alle Stützstellen im Intervall

    # Translation der Stützstellen in [-1,1] auf unser Intervall [0,1]
    count = 1 # Hilfscounter
    for i = 1:N+1:length(xy)-N
        xy[i:i+N] = transformiereKoordinaten(xy_MP[count], Δxy, xy_quad)
        count = count + 1
    end

    return xy

end

#####################################

# Nimmt Vektor u und interpoliert auf gleichmäßig verteilte Stellen
# Verwendet dafür die Visualisierungsmatrix aus Projekt 1
function visualisiereLösung(
    N::Int64,
    N_Q::Int64,
    xy_plot::Array{Float64,1},
    xy::Array{Float64,1},
    w_Bary::Array{Float64,1},
    u::Array{Float64,2})

    # N Polynograd
    # N_Q Zellenanzahl
    # xy_plot Anzahl der Auswertungsstellen nach Transformation
    # xy aktuelle Auswertungsstellen
    # w_Bary baryzentrische Gewichte für die Visualisierungsmatrix
    # u Vektor der interpoliert wird

    u_final1 = zeros(length(xy_plot), N_Q * (N + 1))
    u_output = zeros(length(xy_plot), length(xy_plot))

    #zuerst x-Richtung
    temp = 1
    for i = 1:N+1:length(xy)-N
        if xy[i+N] == xy[end]
            V = Visualisierungsmatrix(xy[i:end], w_Bary, xy_plot[temp:end])
            u_final1[temp:end, :] = V * u[i:end, :]

        else

            for j = temp:length(xy_plot)
                if xy_plot[j] <= xy[i+N] && xy_plot[j+1] > xy[i+N]
                    V = Visualisierungsmatrix(xy[i:i+N],w_Bary,xy_plot[temp:j])
                    u_final1[temp:j, :] = V * u[i:i+N, :]
                    temp = j + 1
                    break
                end
            end
        end
    end

    #u_final1_trans = permutedims(u_final1,(2,1))
    u_final1 = u_final1'

    # Dasselbe Verfahren für die y-Richtung
    temp = 1
    for i = 1:N+1:length(xy)-N
        if xy[i+N] == xy[end]
            V = Visualisierungsmatrix(xy[i:end], w_Bary, xy_plot[temp:end])
            u_output[temp:end, :] = V * u_final1[i:end, :]

        else

            for j = temp:length(xy_plot)
                if xy_plot[j] <= xy[i+N] && xy_plot[j+1] > xy[i+N]
                    V = Visualisierungsmatrix(xy[i:i+N],w_Bary,xy_plot[temp:j])
                    u_output[temp:j, :] = V * u_final1[i:i+N, :]
                    temp = j + 1
                    break
                end
            end
        end
    end

    return u_output

end

############################


function VektorTrixiToMatrixFilter(u::Array{Float64,1},E_Matrix::Array{Int64,2},N::Int64,N_Q::Int64)

    count = 1
    tempKron = spzeros(N_Q,N_Q)
    lsg = zeros(N_Q*(N+1),N_Q*(N+1),4)
    U = reshape(u, 4, Int(length(u) / 4))
    U = convert(Array{Float64,2},U')

    while count <= N_Q^2
        tempShape = reshape(U[1+(count-1)*(N+1)^2:count*(N+1)^2,:],N+1,N+1,4)
        TransShape = permutedims(tempShape,(2,1,3))
        tempIndex = findall(x->x == count,E_Matrix)

        tempKron[tempIndex[1]] = 1
        for i=1:4
            lsg[:,:,i] = lsg[:,:,i] + kron(tempKron,TransShape[:,:,i])
        end

        tempKron[tempIndex[1]] = 0
        tempKron = dropzeros(tempKron)

        count = count + 1
    end

    return lsg
end

########################################

# Wandelt Stützstellen in [-1,1] in Stützstellen in [a,b] um
function transformiereKoordinaten(x_MP::Float64,Δx::Float64,ξ::Array{Float64,1})
    # x_MP: Mittelpunkt einer Zelle
    # Δx: Breite der Zelle
    # ξ: Vektor mit Werten, die transformiert werden sollen
    return x_MP .+ Δx/2*ξ
end

######################################

function Visualisierungsmatrix(x::Array{Float64,1},w::Array{Float64,1},z::Array{Float64,1})
    # x Stützstellen
    # w baryzentrische Gewichte
    # z Auswertungsstellen

    N = length(x)-1
    N_out = length(z)-1

    V = [0.0 for i=1:N_out+1, j=1:N+1]

    for i=1:N_out+1
        check = false

        for j=1:N+1
            if z[i] == x[j]
                V[i,j] = 1
                check = true
            end
        end

        if !check
            summe = 0

            for j=1:N+1
                temp = w[j]/(z[i]-x[j])
                V[i,j] = temp
                summe = summe + temp
            end

            V[i,:] = V[i,:]/summe

        end
    end

    return V
end
