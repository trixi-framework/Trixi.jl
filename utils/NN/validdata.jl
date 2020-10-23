include("approximation.jl")

using HDF5
using Statistics


X = zeros(5,0)
Y = zeros(2,0)

function Output(func, xi, h)
     if func == 1 || func == 2 
      Y = [0; 1]
    elseif func == 3
      if 0 >= xi-(3/2)*h && 0 <= xi+(3/2)*h
        Y = [1; 0]
      else
        Y = [0; 1]
      end
    elseif func == 4
      Y = [1; 0]
    end
    return Y
  end

function addtrainingdata(func, u, a, b, meshsize, polydeg)
    for h in meshsize
        for r in polydeg
            xi = a + (3/2)*h
            left = a+h
            right = a + 2*h
            for i in 1:round((b-a)/h-2)
                Xi = zeros(5,1)
                Yi = zeros(2,1)

                #Create Xi
                _, _, Xi[1] = legendreapprox(u, left-h, left, r)
                Xi[4], Xi[5], Xi[2]= legendreapprox(u, left, right, r)
                _, _, Xi[3] = legendreapprox(u, right, right+1, r)

                #Create label Yi
                Yi = Output(func, xi, h)

                #append to X and Y
                global X = cat(X, Xi, dims=2)
                global Y = cat(Y, Yi, dims=2)

                xi += h
                left += h
                right +=h
            end
        end
    end
    println(size(X))
end


h=0.01:0.01:0.2
r = [1 2 3 4 5 6]

u1(x)=sin(1*pi*x)+sin(2*pi*x)+sin(2*pi*x)+sin(4*pi*x)+sin(5*pi*x)
a = 0
b = 2
addtrainingdata(1, u1, a, b, h, r)

u2(x) =  sin(2*pi*x)*cos(3*pi*x)*sin(4*pi*x)
addtrainingdata(1, u2, a, b, h, r)

a=-1
b=1
u3(x) = sin(pi*x)+exp(x)
addtrainingdata(1, u3, a, b, h, r)


#For u4: just troubled cells
for ul in -20:5:20
    for ur in -20:5:20
        for x0 in -0.75:0.25:0.75
            for h in 0.01:0.01:0.04
                for xi in x0-(3/2)*h:0.1:x0+(3/2)*h
                    for r = [1 2 3 4 5 6]
                        Xi = zeros(5,1)
                        Yi = zeros(2,1)
                        u4(x)=ul*(x<x0) + ur*(x>x0)

                        #Create Xi
                        left = xi - (1/2)*h
                        right= xi + (1/2)*h
                        _, _, Xi[1] = legendreapprox(u4, left-h, left, r)
                        Xi[4], Xi[5], Xi[2]= legendreapprox(u4, left, right, r)
                        _, _, Xi[3] = legendreapprox(u4, right, right+1, r)

                        #Create label Yi
                        Yi = Output(4, xi, h)

                        #append to X and Y
                        global X = cat(X, Xi, dims=2)
                        global Y = cat(Y, Yi, dims=2)

                    end
                end
            end
        end
    end
end
println(size(X))

for i in 1:size(X)[2]
    X[:,i]=X[:,i]./max(maximum(abs.(X[:,i])),1)
end

h5open("validdata.h5", "w") do file
    write(file, "X", X)
    write(file, "Y", Y)
end
