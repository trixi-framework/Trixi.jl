using Tullio: @tullio
using Plots; pyplot()
#include("../../../src/solvers/dg/interpolation.jl")

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

function legendreapprox(u, data, r)
    a = data[:, 1, 1]
    b = data[:, end, end]

    #r=2
    #a=[-0.2, -0.2]
    #b=[0, 0]
    #f(x,y) = sin(2*pi*x)+cos(2*pi*y)

    coef=zeros(r+1,r+1) 
    coef2=zeros(r+1,r+1)  
    help=zeros(r+1,r+1)
    help2=zeros(r+1,r+1)
    f(x, y) = u(x, y)
   
    a1 = a[1]
    b1 = b[1]
    a2 = a[2]
    b2 = b[2]
    h=r+1
    n,w = gauss_lobatto_nodes_weights(h)
  
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
    
    for j=1:r+1
        for i=1:r+1

            for l=1:h #y
                coef[i,j] = 0
                help[i,j] = 0
                for l2=1:h #x
                    poly,_= legendre_polynomial_and_derivative(i-1,n[l2])./sqrt(i-0.5)
                    coef[i,j] += f(nab[l2],nab2[l])*poly*wab[l2]
                    help[i,j] += poly*wab[l2]*poly
                end
                coef[i,j] = coef[i,j]/help[i,j]
                polyj,_= legendre_polynomial_and_derivative(j-1,n[l])./sqrt(j-0.5)
                coef2[i,j] += coef[i,j]*polyj*wab2[l]
                help2[i,j] += polyj*wab2[l]*polyj
            end
            coef2[i,j] = coef2[i,j]/help2[i,j] 

        end
    end


    function uh(x,y)
      uh_approx=0
      for j=1:r+1
        t=y*2/(b2-a2)-(a2+b2)/(b2-a2)
        phi,_= legendre_polynomial_and_derivative(j-1,t)./sqrt(j-0.5)
        sum = 0
        for i=1:r+1
            t2=x*2/(b1-a1)-(a1+b1)/(b1-a1)
            phij,_= legendre_polynomial_and_derivative(i-1,t2)./sqrt(i-0.5)
            sum +=coef2[i,j]*phij
        end
        uh_approx += sum*phi
      end
      uh_approx
    end
    
  
    uhp = uh.(data[1,:,:],data[2,:,:])
    
    #display(plot(xh,yh , f, st=:surface))
    #display(plot(xh,yh, uh, st=:surface))
    
    return uhp
end

