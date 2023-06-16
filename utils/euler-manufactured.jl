#= Usage:

# Start Julia
julia --color=yes

# Install Reduce.jl package (only necessary once)
import Pkg; Pkg.add("Reduce");

# Load file
julia> using Revise
julia> includet("euler-manufactured.jl")

# Run methods to generate source terms. The source terms that need to be
# implemented can be found in the `source_...` variables of the quote'd output.
# If you want, you can also modify, e.g., the ini method to get different manufactured solutions.
julia> euler1d()
julia> euler2d()
julia> euler3d()
=#

using Reduce
@force using Reduce.Algebra

# Original Reduce code (CompressibleEulerEquations 1D)
#=
clear(γ,f,A,ω,c,ini,rho,rho_v1,rho_v2,rho_v3,rho_e,v1,v2,p,x,y,t,u1,u2,u3,u4);

ini := c + A * sin(ω * (x - t));
rho := ini;
rho_v1 := ini;
rho_e := ini^2;

v1 := rho_v1 / rho;
p := (γ - 1) * (rho_e - 1/2 * rho * v1^2);

source_rho    := df(rho, t)    + df(rho_v1, x)
source_rho_v1 := df(rho_v1, t) + df(rho * v1^2 + p, x)
source_rho_e  := df(rho_e, t)  + df((rho_e + p) * v1, x)
=#

function euler1d()
    quote
        ini = c + a * sin(ω * (x - t))
        rho = ini
        rho_v1 = ini
        rho_e = ini^2

        v1 = rho_v1 / rho
        p = (γ - 1) * (rho_e - 1 / 2 * rho * v1^2)

        #! format: off
        source_rho    = df(rho, t)    + df(rho_v1, x)
        source_rho_v1 = df(rho_v1, t) + df(rho * v1^2 + p, x)
        source_rho_e  = df(rho_e, t)  + df((rho_e + p) * v1, x)
        #! format: on
    end |> rcall
end

# Original Reduce code (CompressibleEulerEquations 2D)
#=
clear(γ,f,A,ω,c,ini,rho,rho_v1,rho_v2,rho_v3,rho_e,v1,v2,p,x,y,t,u1,u2,u3,u4);

ini := c + A * sin(ω * (x + y - t));
rho := ini;
rho_v1 := ini;
rho_v2 := ini;
rho_e := ini^2;

v1 := rho_v1 / rho;
v2 := rho_v2 / rho;
p := (γ - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2));

source_rho    := df(rho, t)    + df(rho_v1, x)           + df(rho_v2, y);
source_rho_v1 := df(rho_v1, t) + df(rho * v1^2 + p, x)   + df(rho * v1 * v2, y);
source_rho_v2 := df(rho_v2, t) + df(rho * v1 * v2, x)    + df(rho * v2^2 + p, y);
source_rho_e  := df(rho_e, t)  + df((rho_e + p) * v1, x) + df((rho_e + p) * v2, y);
=#

function euler2d()
    quote
        ini = c + a * sin(ω * (x + y - t))
        rho = ini
        rho_v1 = ini
        rho_v2 = ini
        rho_e = ini^2

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        p = (γ - 1) * (rho_e - 1 / 2 * rho * (v1^2 + v2^2))

        #! format: off
        source_rho    = df(rho, t)    + df(rho_v1, x)           + df(rho_v2, y)
        source_rho_v1 = df(rho_v1, t) + df(rho * v1^2 + p, x)   + df(rho * v1 * v2, y)
        source_rho_v2 = df(rho_v2, t) + df(rho * v1 * v2, x)    + df(rho * v2^2 + p, y)
        source_rho_e  = df(rho_e, t)  + df((rho_e + p) * v1, x) + df((rho_e + p) * v2, y)
        #! format: on
    end |> rcall
end

# Original Reduce code (CompressibleEulerEquations 3D)
#=
clear(γ,f,A,ω,c,a1,a2,a3,ini,rho,rho_v1,rho_v2,rho_v3,rho_e,v1,v2,v3,p,x,y,z,t);

ini := c + A * sin(ω * (x + y + z - t));
rho := ini;
rho_v1 := ini;
rho_v2 := ini;
rho_v3 := ini;
rho_e := ini^2;

v1 := rho_v1 / rho;
v2 := rho_v2 / rho;
v3 := rho_v3 / rho;
p := (γ - 1) * (rho_e - 1/2 * rho * (v1^2 + v2^2 + v3^2));

source_rho    := df(rho, t)    + df(rho_v1, x)           + df(rho_v2, y)           + df(rho_v3, z);
source_rho_v1 := df(rho_v1, t) + df(rho * v1^2 + p, x)   + df(rho * v1 * v2, y)    + df(rho * v1 * v3, z);
source_rho_v2 := df(rho_v2, t) + df(rho * v1 * v2, x)    + df(rho * v2^2 + p, y)   + df(rho * v2 * v3, z);
source_rho_v3 := df(rho_v3, t) + df(rho * v1 * v3, x)    + df(rho * v3 * v3, y)    + df(rho * v3^2 + p, z);
source_rho_e  := df(rho_e, t)  + df((rho_e + p) * v1, x) + df((rho_e + p) * v2, y) + df((rho_e + p) * v3, z);
=#

function euler3d()
    quote
        ini = c + a * sin(ω * (x + y + z - t))
        rho = ini
        rho_v1 = ini
        rho_v2 = ini
        rho_v3 = ini
        rho_e = ini^2

        v1 = rho_v1 / rho
        v2 = rho_v2 / rho
        v3 = rho_v3 / rho
        p = (γ - 1) * (rho_e - 1 / 2 * rho * (v1^2 + v2^2 + v3^2))

        #! format: off
        source_rho    = df(rho, t)    + df(rho_v1, x)           + df(rho_v2, y)           + df(rho_v3, z)
        source_rho_v1 = df(rho_v1, t) + df(rho * v1^2 + p, x)   + df(rho * v1 * v2, y)    + df(rho * v1 * v3, z)
        source_rho_v2 = df(rho_v2, t) + df(rho * v1 * v2, x)    + df(rho * v2^2 + p, y)   + df(rho * v2 * v3, z)
        source_rho_v3 = df(rho_v3, t) + df(rho * v1 * v3, x)    + df(rho * v3 * v3, y)    + df(rho * v3^2 + p, z)
        source_rho_e  = df(rho_e, t)  + df((rho_e + p) * v1, x) + df((rho_e + p) * v2, y) + df((rho_e + p) * v3, z)
        #! format: on
    end |> rcall
end
