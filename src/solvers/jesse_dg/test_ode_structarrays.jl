using OrdinaryDiffEq
using StaticArrays
using StructArrays
using RecursiveArrayTools
using ArrayInterface

# using Logging: global_logger
# using TerminalLoggers: TerminalLogger
# global_logger(TerminalLogger())

function RecursiveArrayTools.recursivecopy(a::AbstractArray{T,N}) where {T<:AbstractArray,N}
    if ArrayInterface.ismutable(a)
      b = similar(a)
      map!(recursivecopy,b,a)
    else
      ArrayInterface.restructure(a,map(recursivecopy,a))
    end
end
ArrayInterface.ismutable(x::StructArray) = true

function rhs!(du::StructArray,u::StructArray,p,t)    
    du .= -u
    return nothing
end
u = StructArray(SVector{2}.(ones(5),2*ones(5)))

# function rhs!(du,u,p,t)    
#   du .= -u
#   return nothing
# end
# u = [ones(2)*i for i = 1:2]

du = similar(u)
prob = ODEProblem(rhs!,u,(0.0,10.))
sol = solve(prob,dt=.01,Euler()) #,progress=true,progress_steps=1)
sol.u[end]

f(x, y) = x
const f1 = x->f(x,1) # anonymous function
f2(x) = f(x,1) 

let x = 1
  @btime f1($x) 
  @btime f2($x) 
end

## closer to real life example
struct Foo end
f(x, y, z::Foo) = x

function time(z)
  f1(y) = let y=y
    x->f(x,y,zz) 
  end
  f2(y) = x->f(x,y,Foo())

  x = 1
  @btime f1($1)($x);
  @btime f2($1)($x); 
end

# actual use case
const eqn = CompressibleEulerEquations1D(1.4)
foo1(orientation) = let equations = eqn
  @inline (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,equations)
end

foo2(orientation) = @inline (uL,uR)->Trixi.flux_chandrashekar(uL,uR,orientation,CompressibleEulerEquations1D(1.4))
let u = (1.,2.,3.)
  @btime foo1($1)($u,$u)
  @btime foo2($1)($u,$u)
end
