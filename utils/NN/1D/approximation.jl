using Tullio: @tullio
include("../../../src/solvers/dg/interpolation.jl")

function legendreapprox(u, a, b, r)
    coef=zeros(r+1)   
    help=zeros(r+1)
    f(x)=u(x)
  
    h=r+1
    #n,w = gauss_nodes_weights(h)
    n,w = gauss_lobatto_nodes_weights(h)
  
    #Transformation: [-1,1] to [a,b]
    nab=zeros(h)
    wab=zeros(h)
    for j=1:h
      nab[j]=(b-a)/2*n[j]+(a+b)/2
      wab[j]=(b-a)/2*w[j]
    end
  
    for i=1:r+1
      for l=1:h
        poly,_= legendre_polynomial_and_derivative(i-1,n[l])./sqrt(i-0.5)
        coef[i] += f(nab[l])*poly*wab[l]
        help[i] += poly*wab[l]*poly
      end
      coef[i] = coef[i]/help[i]
    end
  
    function uh(x)
      uh_approx=0
      for i=1:r+1
        t=x*2/(b-a)-(a+b)/(b-a)
        phi,_= legendre_polynomial_and_derivative(i-1,t)./sqrt(i-0.5)
        uh_approx+=coef[i]*phi
      end
      uh_approx
    end
  
    meanuh = mean(uh,nab)
    
    #xh=a:100:b
    #plotu=uh.(xh)
    #display(plot(xh,plotu,label = "uh"))
    #display(plot!(xh,f,label = "u"))
  
    return uh(a), uh(b), meanuh
  end