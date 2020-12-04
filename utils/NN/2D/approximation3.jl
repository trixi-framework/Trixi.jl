using Tullio: @tullio
#using Plots; pyplot()
include("../../../src/solvers/dg/interpolation.jl")


function legendreapprox(u, data, r)
    a = data[:, 1, 1]
    b = data[:, end, end]

    modal=zeros(r+1,r+1)
    f(x, y) = u(x, y)
   
    a1 = a[1]
    b1 = b[1]
    a2 = a[2]
    b2 = b[2]
    h=r+1
    n,w = gauss_lobatto_nodes_weights(h)
    _, inverse_vandermonde_legendre = vandermonde_legendre(n)

    #Transformation: [-1,1] to [a,b]
    nab=zeros(h)
    nab2=zeros(h)

    for j=1:h
      #x
      nab[j]=(b1-a1)/2*n[j]+(a1+b1)/2
      #y
      nab2[j]=(b2-a2)/2*n[j]+(a2+b2)/2
    end
    indicator=zeros(1,h,h)

    for i in 1:h
        for j in 1:h
            indicator[1,i,j] = u(nab[i],nab2[j])
        end
    end
    
    modal[:,:] = multiply_dimensionwise(inverse_vandermonde_legendre, indicator)
    
    
    return modal[1,1], modal[1,2], modal[2,1]
end

