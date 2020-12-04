using Tullio: @tullio
using Plots; pyplot()
include("../../../src/solvers/dg/interpolation.jl")

function legendre_polynomial_and_derivative(N::Int, x::Real)
    if N == 0
      poly = 1.0
      deriv = 0.0
    elseif N == 1
      poly = convert(Float64, x)
      deriv = 1.0
    else
      poly_Nm2 = 1.0
      poly_Nm1 = convert(Float64, x)
      deriv_Nm2 = 0.0
      deriv_Nm1 = 1.0
  
      poly = 0.0
      deriv = 0.0
      for i in 2:N
        poly = ((2*i-1) * x * poly_Nm1 - (i-1) * poly_Nm2) / i
        deriv=deriv_Nm2 + (2*i-1)*poly_Nm1
        poly_Nm2=poly_Nm1
        poly_Nm1=poly
        deriv_Nm2=deriv_Nm1
        deriv_Nm1=deriv
      end
    end
  
    # Normalize
    poly = poly * sqrt(N+0.5)
    deriv = deriv * sqrt(N+0.5)
  
    return poly, deriv
end

function calc_q_and_l(N::Integer, x::Float64)
    L_Nm2 = 1.0
    L_Nm1 = x
    Lder_Nm2 = 0.0
    Lder_Nm1 = 1.0
  
    local L
    for i in 2:N
      L = ((2 * i - 1) * x * L_Nm1 - (i - 1) * L_Nm2) / i
      Lder = Lder_Nm2 + (2 * i - 1) * L_Nm1
      L_Nm2 = L_Nm1
      L_Nm1 = L
      Lder_Nm2 = Lder_Nm1
      Lder_Nm1 = Lder
    end
  
    q = (2 * N + 1)/(N + 1) * (x * L - L_Nm2)
    qder = (2 * N + 1) * L
  
    return q, qder, L
end
calc_q_and_l(N::Integer, x::Real) = calc_q_and_l(N, convert(Float64, x))


function gauss_lobatto_nodes_weights(n_nodes::Integer)
    # From Kopriva's book
    n_iterations = 10
    tolerance = 1e-15
  
    # Initialize output
    nodes = zeros(n_nodes)
    weights = zeros(n_nodes)
  
    # Get polynomial degree for convenience
    N = n_nodes - 1
  
    # Calculate values at boundary
    nodes[1] = -1.0
    nodes[end] = 1.0
    weights[1] = 2 / (N * (N + 1))
    weights[end] = weights[1]
  
    # Calculate interior values
    if N > 1
      cont1 = pi/N
      cont2 = 3/(8 * N * pi)
  
      # Use symmetry -> only left side is computed
      for i in 1:(div(N + 1, 2) - 1)
        # Calculate node
        # Initial guess for Newton method
        nodes[i+1] = -cos(cont1*(i+0.25) - cont2/(i+0.25))
  
        # Newton iteration to find root of Legendre polynomial (= integration node)
        for k in 0:n_iterations
          q, qder, _ = calc_q_and_l(N, nodes[i+1])
          dx = -q/qder
          nodes[i+1] += dx
          if abs(dx) < tolerance * abs(nodes[i+1])
            break
          end
        end
  
        # Calculate weight
        _, _, L = calc_q_and_l(N, nodes[i+1])
        weights[i+1] = weights[1] / L^2
  
        # Set nodes and weights according to symmetry properties
        nodes[N+1-i] = -nodes[i+1]
        weights[N+1-i] = weights[i+1]
      end
    end
  
    # If odd number of nodes, set center node to origin (= 0.0) and calculate weight
    if n_nodes % 2 == 1
      _, _, L = calc_q_and_l(N, 0)
      nodes[div(N, 2) + 1] = 0.0
      weights[div(N, 2) + 1] = weights[1] / L^2
    end
  
    return nodes, weights
end

# Calculate Lagrange polynomials for a given node distribution.
function lagrange_interpolating_polynomials(x::Float64, nodes, wbary)
    n_nodes = length(nodes)
    polynomials = zeros(n_nodes)
  
    for i in 1:n_nodes
      if isapprox(x, nodes[i], rtol=eps(x))
        polynomials[i] = 1
        return polynomials
      end
    end
  
    for i in 1:n_nodes
      polynomials[i] = wbary[i] / (x - nodes[i])
    end
    total = sum(polynomials)
  
    for i in 1:n_nodes
      polynomials[i] /= total
    end
  
    return polynomials
end

# Calculate the barycentric weights for a given node distribution.
function barycentric_weights(nodes)
    n_nodes = length(nodes)
    weights = ones(n_nodes)
  
    for j = 2:n_nodes, k = 1:(j-1)
      weights[k] *= nodes[k] - nodes[j]
      weights[j] *= nodes[j] - nodes[k]
    end
  
    for j in 1:n_nodes
      weights[j] = 1 / weights[j]
    end
  
    return weights
end


# Calculate Legendre vandermonde matrix and its inverse
function vandermonde_legendre(nodes, N)
  n_nodes = length(nodes)
  n_modes = N + 1
  vandermonde = zeros(n_nodes, n_modes)

  for i in 1:n_nodes
    for m in 1:n_modes
      vandermonde[i, m], _ = legendre_polynomial_and_derivative(m-1, nodes[i])
    end
  end
  # for very high polynomial degree, this is not well conditioned
  inverse_vandermonde = inv(vandermonde)
  return vandermonde, inverse_vandermonde
end
vandermonde_legendre(nodes) = vandermonde_legendre(nodes, length(nodes) - 1)


function multiply_dimensionwise_naive(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 2})
  size_out = size(matrix, 1)
  size_in  = size(matrix, 2)
  n_vars   = size(data_in, 1)
  data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out)

  for i in 1:size_out
    for ii in 1:size_in
      for v in 1:n_vars
        data_out[v, i] += matrix[i, ii] * data_in[v, ii]
      end
    end
  end

  return data_out
end

function multiply_dimensionwise(matrix::AbstractMatrix, data_in::AbstractArray{<:Any, 3})
  # 2D
  # optimized version of multiply_dimensionwise_naive
  size_out = size(matrix, 1)
  n_vars   = size(data_in, 1)
  data_out = zeros(promote_type(eltype(data_in), eltype(matrix)), n_vars, size_out, size_out)

  multiply_dimensionwise!(data_out, matrix, data_in)

  return data_out
end

function multiply_dimensionwise!(data_out::AbstractArray{<:Any, 3}, matrix::AbstractMatrix,
  data_in:: AbstractArray{<:Any, 3},
  tmp1=zeros(eltype(data_out), size(data_out, 1), size(matrix, 1), size(matrix, 2)))

# Interpolate in x-direction
@tullio threads=false tmp1[v, i, j]     = matrix[i, ii] * data_in[v, ii, j]

# Interpolate in y-direction
@tullio threads=false data_out[v, i, j] = matrix[j, jj] * tmp1[v, i, jj]

return nothing
end


function legendreapprox(u, data, r)
    a = data[:, 1, 1]
    b = data[:, end, end]

    #r = 3
    #a = [-1, -1]
    #b = [1, 1]
    #f(x,y) = sin(2*pi*x)+cos(2*pi*y)

    coef=zeros(r+1,r+1) 
    coef2=zeros(r+1,r+1)  
    help=zeros(r+1,r+1)
    help2=zeros(r+1,r+1)
    modal=zeros(r+1,r+1)
    f(x, y) = u(x, y)
   
    a1 = a[1]
    b1 = b[1]
    a2 = a[2]
    b2 = b[2]
    h=r+1
    n,w = gauss_lobatto_nodes_weights(h)
    _, inverse_vandermonde_legendre = vandermonde_legendre(n)
    wbary = barycentric_weights(n)
    polynomials2 = zeros(h,h)
    
    #println(size(polynomials2))

    #Transformation: [-1,1] to [a,b]
    nab=zeros(h)
    wab=zeros(h)
    nab2=zeros(h)
    wab2=zeros(h)
    for j=1:h
      #x
      nab[j]=(b1-a1)/2*n[j]+(a1+b1)/2
      wab[j]=(b1-a1)/2*w[j]
      #y
      nab2[j]=(b2-a2)/2*n[j]+(a2+b2)/2
      wab2[j]=(b2-a2)/2*w[j]
    end
    for i in 1:h
      polynomials2[:,i] = lagrange_interpolating_polynomials(n[i], n, wbary)
    end
    
    for j=1:h
        for i=1:h

            for l=1:h #y
                coef[i,j] = 0
                help[i,j] = 0
                for l2=1:h #x
                    coef[i,j] += f(nab[l2],nab2[l])*wab[l2]*polynomials2[i,l2]
                    help[i,j] += polynomials2[i,l2]*wab[l2]*polynomials2[i,l2]
                end
                coef[i,j] = coef[i,j]/help[i,j]
                coef2[i,j] += coef[i,j]*wab2[l]*polynomials2[j,l]
                help2[i,j] += polynomials2[j,l]*wab2[l]*polynomials2[j,l]
            end
            coef2[i,j] = coef2[i,j]/help2[i,j] 

        end
    end

    
    function uh(x,y)
      uh_approx=0
      for j=1:r+1
        t=y*2/(b2-a2)-(a2+b2)/(b2-a2)
        phi= lagrange_interpolating_polynomials(t, n, wbary)
        sum = 0
        for i=1:r+1
            t2=x*2/(b1-a1)-(a1+b1)/(b1-a1)
            phii = lagrange_interpolating_polynomials(t2, n, wbary)
            sum +=coef2[i,j]*phii[i]
        end
        uh_approx += sum*phi[j]
      end
      uh_approx
    end
    
    
  
    #uhp = uh.(data[1,:,:],data[2,:,:])
    indicator=zeros(1,h,h)
    indicator[1,:,:]=coef2[:,:]
    
    modal[:,:] = multiply_dimensionwise(inverse_vandermonde_legendre, indicator)
    
    return modal[:,:] #modal[1,1], modal[1,2], modal[2,1] #coef2[1,1], coef2[1,2], coef2[2,1]   #uhp
end


r = 4
n_nodes = r + 1
n, _ = gauss_lobatto_nodes_weights(n_nodes)
_, inverse_vandermonde_legendre = vandermonde_legendre(n)
data = zeros(2,r+1,r+1)
u(x,y) = sin(2*x*pi)+cos(2*pi*y)  #-4*(x>-0.97)+2(x<-0.97)

data[1,:,:] = [-1.0 -1.0 -1.0 -1.0 -1.0; -0.9892079272096242 -0.9892079272096242 -0.9892079272096242 -0.9892079272096242 -0.9892079272096242; -0.96875 -0.96875 -0.96875 -0.96875 -0.96875; -0.9482920727903758 -0.9482920727903758 -0.9482920727903758 -0.9482920727903758 -0.9482920727903758; -0.9375 -0.9375 -0.9375 -0.9375 -0.9375]
data[2,:,:] = [-1.0 -0.9892079272096242 -0.96875 -0.9482920727903758 -0.9375; -1.0 -0.9892079272096242 -0.96875 -0.9482920727903758 -0.9375; -1.0 -0.9892079272096242 -0.96875 -0.9482920727903758 -0.9375; -1.0 -0.9892079272096242 -0.96875 -0.9482920727903758 -0.9375; -1.0 -0.9892079272096242 -0.96875 -0.9482920727903758 -0.9375]



indicator = zeros(1,r+1,r+1)
indicator2 = zeros(r+1,r+1)
indicator3 = zeros(r+1,r+1)
modal = zeros(1, r+1,r+1)
modal2 = zeros(1,r+1,r+1)
modal3 = zeros(1,r+1,r+1)

#data[1,:,:] = [-1.0 -1.0 -1.0 -1.0; -0.9654508497187474 -0.9654508497187474 -0.9654508497187474 -0.9654508497187474; -0.9095491502812526 -0.9095491502812526 -0.9095491502812526 -0.9095491502812526; -0.875 -0.875 -0.875 -0.875]
#data[2,:,:] = [-1.0 -0.9654508497187474 -0.9095491502812526 -0.875; -1.0 -0.9654508497187474 -0.9095491502812526 -0.875; -1.0 -0.9654508497187474 -0.9095491502812526 -0.875; -1.0 -0.9654508497187474 -0.9095491502812526 -0.875]
#indicator[ :, :] = [-10.791241160033294 2.1880467332628015 -0.49877185204354624 5.855733717364666; 2.1880467332608586 -0.40950662810862437 0.18319673146139093 -0.9485745167099635; -0.4987718520415141 0.1831967314610743 0.17527936003660333 0.5196030700903376; 5.855733717367384 -0.9485745167117408 0.5196030700914045 -1.3719687808129981]
#modal[1,:,:]= [0.2146589820991673 0.06353698181980239 0.1220098149458967 -0.07850796538117225; 0.06353698181974901 -0.3456697190162309 0.40370113841587413 -0.5693188479023665; 0.1220098149457476 0.4037011384153361 -0.19723494308766906 0.548297486557302; -0.07850796538106387 -0.5693188479015346 0.5482974865573766 -0.8117765857984287]

indicator[1, :, :]=legendreapprox(u, data , r)
#modal[1,:,:] = multiply_dimensionwise(inverse_vandermonde_legendre, indicator)
#modal3[1,:,:] = multiply_dimensionwise_naive(inverse_vandermonde_legendre, indicator3)



display(plot(data[1,:,:],data[2,:,:],indicator[1,:,:], st=:surface))
#display(plot(data[1,:,:],data[2,:,:], modal[1,:,:], st=:surface))

#display(plot(data[1,:,:],data[2,:,:],indicator2, st=:surface))
#display(plot(data[1,:,:],data[2,:,:], modal2[1,:,:], st=:surface))

#display(plot(data[1,:,:],data[2,:,:],indicator3, st=:surface))
#display(plot(data[1,:,:],data[2,:,:], modal3[1,:,:], st=:surface))



