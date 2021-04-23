using Octavian 
using LinearAlgebra

A = randn(8,8)
x = randn(8,10000)
b = similar(x)
f(x) = exp(x+1) + sin(x)

function tmap!(f,out,x)
    Threads.@threads for i = 1:length(x)
        out[i] = f(x[i])
    end
end

@btime matmul!($b,$A,$x) # 4.702 μs (0 allocations: 0 bytes)
@btime tmap!($f,$b,$x)   # 195.632 μs (41 allocations: 3.16 KiB)
@btime map!($f,$b,$x)    # 1.126 ms (0 allocations: 0 bytes)

function time1(b,A,x) 
    matmul!(b,A,x) 
    tmap!(f,b,x)   
end
function time2(b,A,x) 
    mul!(b,A,x) 
    tmap!(f,b,x) 
end
function time3(b,A,x) 
    mul!(b,A,x) 
    map!(f,b,x) 
end

@btime time1($b,$A,$x) # 42.934 ms (42 allocations: 3.19 KiB)
@btime time2($b,$A,$x) # 323.995 μs (41 allocations: 3.16 KiB)
@btime time3($b,$A,$x) # 1.404 ms (0 allocations: 0 bytes)