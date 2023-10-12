# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    JinXinCompressibleEulerEquations2D{RealT <: Real} <:
       AbstractJinXinEquations{2, 12}
"""
struct JinXinCompressibleEulerEquations2D{RealT <: Real,CollisionOp} <:                                   
       AbstractJinXinEquations{2, 12}
    eps_relaxation    ::RealT    # relaxation parameter epsilon with the tendency epsilon -> 0
    eps_relaxation_inv::RealT    # relaxation parameter epsilon with the tendency epsilon -> 0

    a::SVector{4, RealT} # discrete molecular velocity components in x-direction
    sqrt_a::SVector{4, RealT} # discrete molecular velocity components in x-direction
    sqrt_a_inv::SVector{4, RealT} # discrete molecular velocity components in x-direction
    b::SVector{4, RealT} # discrete molecular velocity components in y-direction
    sqrt_b::SVector{4, RealT} # discrete molecular velocity components in x-direction
    sqrt_b_inv::SVector{4, RealT} # discrete molecular velocity components in x-direction
    equations_relaxation::CompressibleEulerEquations2D{RealT}
    collision_op::CollisionOp   # collision operator for the collision kernel
end

function JinXinCompressibleEulerEquations2D(eps_relaxation,a1,a2,a3,a4,b1,b2,b3,b4,equations_relaxation)
    
    a   = SVector(a1,a2,a3,a4)
    sa  = SVector(sqrt(a1),sqrt(a2),sqrt(a3),sqrt(a4))
    sai = SVector(1.0/sqrt(a1),1.0/sqrt(a2),1.0/sqrt(a3),1.0/sqrt(a4))

    b   = SVector(b1,b2,b3,b4)
    sb  = SVector(sqrt(b1),sqrt(b2),sqrt(b3),sqrt(b4))
    sbi = SVector(1.0/sqrt(b1),1.0/sqrt(b2),1.0/sqrt(b3),1.0/sqrt(b4))

    collision_op = collision_bgk

    JinXinCompressibleEulerEquations2D(eps_relaxation, 1.0/eps_relaxation, a, sa, sai, b, sb, sbi, equations_relaxation, collision_op)
end

function varnames(::typeof(cons2cons), equations::JinXinCompressibleEulerEquations2D)
    ("rho", "rho_v1", "rho_v2", "rho_e", "vu1", "vu2", "vu3", "vu4", "wu1", "wu2", "wu3", "wu4")   
end
function varnames(::typeof(cons2prim), equations::JinXinCompressibleEulerEquations2D)
    varnames(cons2cons, equations)
end

# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::JinXinCompressibleEulerEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::JinXinCompressibleEulerEquations2D)
    rho = 1.0                                                                                       
    rho_v1 = 0.1                                                                                    
    rho_v2 = -0.2                                                                                   
    rho_e = 10.0                                                                                    

    u_cons = (rho, rho_v1, rho_v2, rho_e)
    eq_relax = equations.equations_relaxation
    vu = flux(u_cons,1,eq_relax)
    wu = flux(u_cons,2,eq_relax)

    return SVector(rho, rho_v1, rho_v2, rho_e, vu[1], vu[2], vu[3], vu[4], wu[1], wu[2], wu[3], wu[4])  
end

struct InitialConditionJinXin{IC}
  initial_condition::IC
end

@inline function (ic::InitialConditionJinXin)(x,t,equations)

    eq_relax = equations.equations_relaxation
    u = ic.initial_condition(x,t,eq_relax)
    vu = flux(u,1,eq_relax)
    wu = flux(u,2,eq_relax)
    
    return SVector(u[1], u[2], u[3], u[4], vu[1], vu[2], vu[3], vu[4], wu[1], wu[2], wu[3], wu[4])
end

# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::JinXinCompressibleEulerEquations2D)
@inline function source_terms_JinXin_Relaxation(u, x, t,                                             
                                               equations::JinXinCompressibleEulerEquations2D)             

    # relaxation parameter
    eps_inv= equations.eps_relaxation_inv

    # compute compressible Euler fluxes
    u_cons = SVector(u[1], u[2], u[3], u[4])
    eq_relax = equations.equations_relaxation
    vu = flux(u_cons,1,eq_relax)
    wu = flux(u_cons,2,eq_relax)

    # compute relaxation terms
    du1 = 0.0
    du2 = 0.0
    du3 = 0.0
    du4 = 0.0
    du5 = -eps_inv * (u[5] - vu[1])
    du6 = -eps_inv * (u[6] - vu[2])
    du7 = -eps_inv * (u[7] - vu[3])
    du8 = -eps_inv * (u[8] - vu[4])
    du9 = -eps_inv * (u[9] - wu[1])
    du10= -eps_inv * (u[10]- wu[2])
    du11= -eps_inv * (u[11]- wu[3])
    du12= -eps_inv * (u[12]- wu[4])
                                                                                                    
    return SVector(du1, du2, du3, du4, du5, du6, du7, du8, du9, du10, du11, du12)                                                              
end   


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::JinXinCompressibleEulerEquations2D)
    if orientation == 1
      a = equations.a
      flux = SVector(u[5],u[6],u[7],u[8],a[1]*u[1],a[2]*u[2],a[3]*u[3],a[4]*u[4],0,0,0,0)
    else
      b = equations.b
      flux = SVector(u[9],u[10],u[11],u[12],0,0,0,0,b[1]*u[1],b[2]*u[2],b[3]*u[3],b[4]*u[4])
    end
    return flux
end


@inline function flux_upwind(u_ll, u_rr, orientation::Integer,
                              equations::JinXinCompressibleEulerEquations2D)
    if orientation == 1
      sai = equations.sqrt_a_inv
      sa  = equations.sqrt_a
      #dissipation = SVector(sai[1]*(u_rr[5] - u_ll[5]), sai[2]*(u_rr[6] - u_ll[6]), sai[3]*(u_rr[7] - u_ll[7]), sai[4]*(u_rr[8] - u_ll[8]), sa[1]*(u_rr[1] - u_ll[1]), sa[2]*(u_rr[2] - u_ll[2]), sa[3]*(u_rr[3] - u_ll[3]), sa[4]*(u_rr[4] - u_ll[4]),0.0,0.0,0.0,0.0) 
      dissipation = SVector(sa[1]*(u_rr[1] - u_ll[1]), sa[2]*(u_rr[2] - u_ll[2]), sa[3]*(u_rr[3] - u_ll[3]), sa[4]*(u_rr[4] - u_ll[4]), sa[1]*(u_rr[5] - u_ll[5]), sa[2]*(u_rr[6] - u_ll[6]), sa[3]*(u_rr[7] - u_ll[7]), sa[4]*(u_rr[8] - u_ll[8]),0.0,0.0,0.0,0.0) 
    else
      sbi = equations.sqrt_b_inv
      sb  = equations.sqrt_b
      #dissipation = SVector(sbi[1]*(u_rr[9] - u_ll[9]), sbi[2]*(u_rr[10] - u_ll[10]), sbi[3]*(u_rr[11] - u_ll[11]), sbi[4]*(u_rr[12] - u_ll[12]),0.0,0.0,0.0,0.0, sb[1]*(u_rr[1] - u_ll[1]), sb[2]*(u_rr[2] - u_ll[2]), sb[3]*(u_rr[3] - u_ll[3]), sb[4]*(u_rr[4] - u_ll[4])) 
      dissipation = SVector(sb[1]*(u_rr[1] - u_ll[1]), sb[2]*(u_rr[2] - u_ll[2]), sb[3]*(u_rr[3] - u_ll[3]), sb[4]*(u_rr[4] - u_ll[4]),0.0,0.0,0.0,0.0,sb[1]*(u_rr[9] - u_ll[9]), sb[2]*(u_rr[10] - u_ll[10]), sb[3]*(u_rr[11] - u_ll[11]), sb[4]*(u_rr[12] - u_ll[12]))
    end
    return 0.5 * (flux(u_ll,orientation,equations) + flux(u_rr,orientation,equations) - dissipation)
end


"""                                                                                                 
    collision_bgk(u, dt, equations::LatticeBoltzmannEquations2D)                                    
                                                                                                    
Collision operator for the Bhatnagar, Gross, and Krook (BGK) model.                                 
"""                                                                                                 
@inline function collision_bgk(u, dt, equations::JinXinCompressibleEulerEquations2D)                       
    # relaxation parameter
    eps = equations.eps_relaxation

    # compute compressible Euler fluxes
    u_cons = SVector(u[1], u[2], u[3], u[4])
    eq_relax = equations.equations_relaxation
    vu = flux(u_cons,1,eq_relax)
    wu = flux(u_cons,2,eq_relax)

    dt_ = dt * 1.0
    factor =1.0/ (eps + dt_)

    # compute relaxation terms
    du1 = 0.0
    du2 = 0.0
    du3 = 0.0
    du4 = 0.0
    du5 = -u[5] + factor * (eps * u[5] + dt_ * vu[1])
    du6 = -u[6] + factor * (eps * u[6] + dt_ * vu[2])
    du7 = -u[7] + factor * (eps * u[7] + dt_ * vu[3])
    du8 = -u[8] + factor * (eps * u[8] + dt_ * vu[4])
    du9 = -u[9] + factor * (eps * u[9] + dt_ * wu[1])
    du10= -u[10]+ factor * (eps * u[10]+ dt_ * wu[2])
    du11= -u[11]+ factor * (eps * u[11]+ dt_ * wu[3])
    du12= -u[12]+ factor * (eps * u[12]+ dt_ * wu[4])
                                                                                                    
    return SVector(du1, du2, du3, du4, du5, du6, du7, du8, du9, du10, du11, du12)                                                              
end  



@inline function max_abs_speeds(u,equations::JinXinCompressibleEulerEquations2D)
    sa = equations.sqrt_a
    sb = equations.sqrt_b

    return max(sa[1],sa[2],sa[3],sa[4]), max(sb[1],sb[2],sb[3],sb[4])
end

# not correct yet!!
# Convert conservative variables to primitive
@inline cons2prim(u, equations::JinXinCompressibleEulerEquations2D) = u

# Convert conservative variables to entropy variables
@inline cons2entropy(u, equations::JinXinCompressibleEulerEquations2D) = u
end # @muladd
