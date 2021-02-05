#include("fv_dg_2d.jl")

@inline function reconstruction_large_stencil(u_mm,u_ll,u_rr,u_pp,x_mm,x_ll,x_rr,x_pp,x_interface,index,limiter,dg,equations)
  if (index==2)
    ux_ll1 = 1.0*(u_rr - u_mm) / (x_rr - x_mm + eps(x_rr))
  else
    ux_ll1 = (u_ll - u_mm) / (x_ll - x_mm + eps(x_ll))
  end
  ux_ll2 = (u_rr - u_ll) / (x_rr-x_ll + eps(x_rr))
  ux_ll = limiter.(ux_ll1,ux_ll2)

  ux_rr1 = (u_rr - u_ll) / (x_rr-x_ll + eps(x_rr))
  if (index==nnodes(dg))
    ux_rr2 = 1.0*(u_pp - u_ll) / (x_pp - x_ll + eps(x_rr))
  else
    ux_rr2 = (u_pp - u_rr) / (x_pp - x_rr + eps(x_rr))
  end
  ux_rr = limiter.(ux_rr1,ux_rr2)

  u_ll = choose_positive_value(u_ll + ux_ll * (x_interface[index-1] - x_ll),u_ll,equations)
  u_rr = choose_positive_value(u_rr + ux_rr * (x_interface[index-1] - x_rr),u_rr,equations)
  return u_ll,u_rr
end

@inline function reconstruction_large_irregular(u_mm,u_ll,u_rr,u_pp,x_mm,x_ll,x_rr,x_pp,x_interface,index,limiter,dg,equations)
  if (index==2)
    ux_ll1 = (u_rr - u_mm) / (x_rr - x_mm + eps(x_rr))
  else
    ux_ll1 = 2 * (u_ll - u_mm) / (x_rr - x_mm + eps(x_ll))
  end
  ux_ll2 =   2 * (u_rr - u_ll) / (x_rr-x_mm + eps(x_rr))
  ux_ll = limiter.(ux_ll1,ux_ll2)

  ux_rr1 = 2 * (u_rr - u_ll) / (x_pp - x_ll + eps(x_rr))
  if (index==nnodes(dg))
  else
    ux_rr2 = (u_pp - u_ll) / (x_pp - x_ll + eps(x_rr))
    ux_rr2 = 2*(u_pp - u_rr) / (x_pp - x_ll + eps(x_rr))
  end
  ux_rr = limiter.(ux_rr1,ux_rr2)

  u_ll = choose_positive_value(u_ll + ux_ll*(x_interface[index-1]-x_ll),u_ll,equations)
  u_rr = choose_positive_value(u_rr + ux_rr*(x_interface[index-1]-x_rr),u_rr,equations)
  return u_ll,u_rr
end

@inline function reconstruction_small_stencil(u_mm,u_ll,u_rr,u_pp,x_mm,x_ll,x_rr,x_pp,x_interface,index,limiter,dg,equations)
  ux_ll1 = (u_ll - u_mm)/(x_ll - x_mm + eps(x_ll))
  ux_ll2 = (u_rr - u_ll)/(x_rr-x_ll + eps(x_rr))
  ux_ll = limiter.(ux_ll1,ux_ll2)

  ux_rr1 = (u_rr - u_ll)/(x_rr-x_ll + eps(x_rr))
  ux_rr2 = (u_pp - u_rr)/(x_pp - x_rr + eps(x_rr))
  ux_rr = limiter.(ux_rr1,ux_rr2)

  u_ll = choose_positive_value(u_ll + ux_ll*(x_interface[index-1]-x_ll),u_ll,equations)
  u_rr = choose_positive_value(u_rr + ux_rr*(x_interface[index-1]-x_rr),u_rr,equations)
  return u_ll,u_rr
end

@inline function reconstruction_O1(u_mm,u_ll,u_rr,u_pp,x_mm,x_ll,x_rr,x_pp,x_interface,index,limiter,dg,equations)
  return u_ll,u_rr
end

@inline function choose_positive_value(u,u_safe,equations::CompressibleEulerEquations2D)
  if (u[1]<0.0)||(u[4]<0.0)
    u_positive = u_safe
  else
    u_positive = u
  end
  return u_positive
end

@inline function minmod(sl,sr)
   s = 0.0
   if sign(sl)==sign(sr)
     s = sign(sl)*min(abs(sl),abs(sr))
   end
  return s
end

@inline function central_recon(sl,sr)
    s = 0.5*(sl+sr)
  return s
end

@inline function no_recon(sl,sr)
    s = false*sl
  return s
end

@inline function monotonized_central(sl,sr)
   s = 0.0
   if sign(sl)==sign(sr)
     s = sign(sl)*min(2*abs(sl),2*abs(sr),0.5*abs(sl+sr))
   end
  return s
end
