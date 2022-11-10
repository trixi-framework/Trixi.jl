using Plots
using LinearAlgebra

x = Vector(LinRange(0,10,100))
y = Vector(LinRange(0,10,100))

z = zeros(100,100)

for i = 1:100, j = 1:100
    if norm([x[i], y[j]] - [5,5]) < 2
        z[i,j] = 3 + cos((x[i]-5)/4*pi)*cos((y[j]-5)/4*pi)
    else
        z[i,j] = 3
    end
end

pyplot()
surface(x,y,z)