struct PerturbationMomentSystem1D{RealT<:Real} <: AbstractPerturbationMomentSystem{1, 9} 
    vxr::RealT
    theta_r::RealT
    tau::RealT
  end
  
  varnames(::typeof(cons2prim), ::PerturbationMomentSystem1D) = ("rho", "vx", "theta")
  varnames(::typeof(cons2cons), ::PerturbationMomentSystem1D) = ("w0", "w0x", "w0y", "w1", "w0xx", "w0yy", "w0xy", "w1x", "w1y")
  
  @inline function flux(u, orientation::Integer, equations::PerturbationMomentSystem1D)
    w0, w0x, w0y, w1, w0xx, w0yy, w0xy, w1x, w1y = u

  
    @unpack vxr, theta_r = equations
 

      f1  = vxr * w0  + sqrt(theta_r) * w0x
      f2  = vxr * w0x + sqrt(theta_r) * (w0 - w1) + 2.0 * sqrt(theta_r) * w0xx
      f3  = vxr * w0y + 2.0 * sqrt(theta_r) * w0xy 
      f4  = vxr * w1 + sqrt(theta_r) * (5.0 * w1x - 2.0 * w0x)/3.0
      f5  = vxr * w0xx + 2.0 * sqrt(theta_r)* (w0x - w1x)/3.0 
      f6  = vxr * w0yy + sqrt(theta_r) * (w1x - w0x)/3.0
      f7  = vxr * w0xy + sqrt(theta_r) * (w0y - w1y)/2.0 
      f8  = vxr * w1x + sqrt(theta_r) * (w1 - 4.0 * w0xx/5.0)
      f9  = vxr * w1y - 4.0 * sqrt(theta_r) * w0xy / 5.0
    
    return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
  end
  
  
  @inline function cons2prim(u, equations::PerturbationMomentSystem1D)
    w0, w0x, w0y, w1, w0xx, w0yy, w0xy, w1x, w1y = u
    @unpack vxr, theta_r = equations
  
  
    rho_r = 2.0
    vyr = 0.0

    
    rho = w0 * rho_r
    v_x = vxr + w0x * sqrt(theta_r) / w0
    v_y = vyr + w0y * sqrt(theta_r) / w0
    theta = theta_r - (w0x^2 * theta_r)/(3 * w0^2) - (w0y^2 * theta_r)/(3 * w0^2) - (w1 * theta_r)/ w0
    
    # sigma_xx = - (2 * w0x^2 * theta_r * rho_r)/(3 * w0) + 2*w0xx * theta_r * rho_r + (w0y^2 * theta_r * rho_r)/(3 * w0)
    # sigma_xy = 2 * w0xy * theta_r * rho_r - (w0x * w0y * theta_r * rho_r)/w0
    # sigma_yy = (w0x^2 * theta_r * rho_r)/(3 * w0) - (2*w0y^2*theta_r*rho_r)/(3*w0)+2*w0yy*theta_r*rho_r
    # q_x = (w0x^3 * sqrt(theta_r)^3 * rho_r)/ w0^2 - (2*w0x*w0xx*sqrt(theta_r)^3*rho_r)/w0 - (2*w0xy*w0y*sqrt(theta_r)^3*rho_r)/w0 + (w0x*w0y^2*sqrt(theta_r)^3 * rho_r)/w0^2 + (5 * w0x *w1 *sqrt(theta_r)^3*rho_r)/(2*w0) - (5*w1x*sqrt(theta_r)^3*rho_r)/2
    # q_y = - (2*w0x * w0xy * sqrt(theta_r)^3*rho_r)/w0 + (w0x^2 * w0y * sqrt(theta_r)^3 * rho_r)/w0^2 + (w0y^3*sqrt(theta_r)^3 * rho_r)/w0^2 - (2*w0y *w0yy* sqrt(theta_r)^3 * rho_r)/w0 + (5*w0y*w1*sqrt(theta_r)^3*rho_r)/2*w0 - (5*w1y*sqrt(theta_r)^3*rho_r)/2

    
    return SVector(rho, v_x, theta)
  end

  
  @inline function cons2entropy(u, equations::PerturbationMomentSystem1D)
    w0, w0x, w0y, w1, w0xx, w0yy, w0xy, w1x, w1y = u
  
    return SVector(w0, w0x, w0y, w1, w0xx, w0yy, w0xy, w1x, w1y)
  end
  
  @inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::PerturbationMomentSystem1D)
    @unpack vxr, theta_r = equations
    ab1 = abs(vxr) + 2.0 * sqrt(5 * theta_r / 3)

    return ab1
  end
  
  @inline function max_abs_speeds(u, equations::PerturbationMomentSystem1D)
    @unpack vxr, theta_r = equations
    ab1 = abs(vxr) + 2.0 * sqrt(5 * theta_r / 3)

    return ab1
  end



  function initial_condition_convergence_test(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack vxr = equations 
    c = 2
    A = 0.1
    L = 2
    f = 1/L
    ω = 2 * pi * f
    x1, = x
    ini = c + A * sin(ω * (x[1] - t))


    rho = ini
    rho_r = 2
    drho = rho - rho_r

    vx = 1
  
    dv_x = vx - vxr
    dv_y = 0
    theta = 2 * (ini - 0.5)/3
    theta_r = 2/3
    dtheta = theta - theta_r
    R = 8.314
 
    sigma_xx = (ini-2)/100000
    sigma_xy = 0
    sigma_yy = 0
    q_x = (ini-2)/100000
    q_y = 0

    w0 = rho/rho_r
    w0x = (rho * dv_x)/(rho_r * sqrt(theta_r))
    w0y = 0
    w1 = - (dtheta * rho)/(rho_r * theta_r) - rho*dv_x^2/(3 * rho_r * theta_r) 
    w0xx = (sigma_xx +drho * dv_x^2 * 2 /3)/(2 * rho_r * theta_r)+(dv_x^2 * 2 /3)/(2 * theta_r)
    w0yy = -(rho * dv_x^2 /3)/(2 * theta_r * rho_r)
    w0xy = 0
    w1x =  - 2.0 * q_x / (5.0* rho_r * sqrt(theta_r).^3.0) - (2.0 * (sigma_xx * dv_x + sigma_xy * dv_y))/(5.0*rho_r* sqrt(theta_r).^3.0) - (dtheta * dv_x * rho)/ (rho_r * sqrt(theta_r).^3.0) - rho * (dv_x * dv_y^2 + dv_x^3)/(5.0 * rho_r * sqrt(theta_r).^3.0)
    w1y =  - 2.0 * q_y / (5.0* rho_r * sqrt(theta_r).^3.0) - (2.0 * (sigma_xy * dv_x + sigma_yy * dv_y))/(5.0 * rho_r * sqrt(theta_r).^3.0) - (dtheta * dv_y * rho)/ (rho_r * sqrt(theta_r).^3.0) - rho * (dv_y * dv_x^2 + dv_y^3)/(5.0 * rho_r * sqrt(theta_r).^3.0) 
    return SVector(w0, w0x, w0y, w1, w0xx, w0yy, w0xy, w1x, w1y)
  end


  # @inline function source_terms_convergence_test(u, x, t, equations::PerturbationMomentSystem1D)
  #   w0, w0x, w0y, w1, w0xx, w0yy, w0xy, w1x, w1y = u
  #   tau  = calc_tau(equations)
  
  #   du1 = -w0 * w0xx + w0x * w0x/ 3.0 - w0y * w0y/ 6.0 
  #   du2 = -w0 * w0yy - w0x * w0x/ 6.0 + w0y * w0y/ 3.0
  #   du3 = -w0 * w0xy + w0x * w0y/ 4.0 + w0y * w0x/ 4.0
  #   du4 = 2.0 * w1 * w0x / 3.0 + 4.0 * w0x * w0xx/15.0 + 4.0 * w0y * w0xy/ 15.0 - 2.0 * w0 * w1x / 3.0
  #   du5 = 2.0 * w1 * w0y / 3.0 + 4.0 * w0y * w0yy/ 15.0 + 4.0 * w0x * w0xy/15.0 - 2.0 * w0 * w1y / 3.0
      
  #   return SVector(0.0, 0.0, 0.0, 0.0, du1/tau, du2/tau, du3/tau, du4/tau, du5/tau)

  # end


  @inline function source_terms_convergence_test(u, x, t, equations::PerturbationMomentSystem1D)
    
    @unpack vxr = equations 
    x1, = x
    c = 2
    L = 2
    f = 1/L
    ω = 2 * pi * f
    A = 0.1
    R = 8.314
    tau = calc_tau(equations)
    z = t
      
    w0, w0x, w0y, w1, w0xx, w0yy, w0xy, w1x, w1y = u

    a1 = -w0 * w0xx + w0x * w0x/ 3.0 - w0y * w0y/ 6.0 
    a2 = -w0 * w0yy - w0x * w0x/ 6.0 + w0y * w0y/ 3.0
    a3 = -w0 * w0xy + w0x * w0y/ 4.0 + w0y * w0x/ 4.0
    a4 = 2.0 * w1 * w0x / 3.0 + 4.0 * w0x * w0xx/15.0 + 4.0 * w0y * w0xy/ 15.0 - 2.0 * w0 * w1x / 3.0
    a5 = 2.0 * w1 * w0y / 3.0 + 4.0 * w0y * w0yy/ 15.0 + 4.0 * w0x * w0xy/15.0 - 2.0 * w0 * w1y / 3.0

    #  a1 = 0
    #  a2 = 0
    #  a3 = 0
    #  a4 = 0
    #  a5 = 0

    du1 = 0
    du2 =(625*2^(13/2)*ω*cos((x1-z)*ω)*sin((x1-z)*ω)+700003*sqrt(2)*ω*cos((x1-z)*ω))/(4000000*sqrt(3))
    du3 = 0
    du4 = ((40000*vxr-40000)*ω*cos((x1-z)*ω)*sin((x1-z)*ω)+(700003*vxr-700006)*ω*cos((x1-z)*ω))/6000000
    du5 = -((800000*vxr-800000)*ω*cos((x1-z)*ω)*sin((x1-z)*ω)+(-1800000*vxr^3+5400000*vxr^2+8599979*vxr-12200003)*ω*cos((x1-z)*ω))/120000000 + a1/tau
    du6 = ((100000*vxr-100000)*ω*cos((x1-z)*ω)*sin((x1-z)*ω)+(-225000*vxr^3+675000*vxr^2+1075003*vxr-1525006)*ω*cos((x1-z)*ω))/30000000 + a2/tau
    du7 = a3/tau 
    du8 = ((9375*2^(11/2)*vxr^2-9375*2^(13/2)*vxr+3125*2^(11/2))*ω*cos((x1-z)*ω)*sin((x1-z)*ω)+(28125*2^(9/2)*vxr^4-28125*2^(13/2)*vxr^3+5550009*sqrt(2)*vxr^2-7500027*sqrt(2)*vxr+200003*2^(5/2))*ω*cos((x1-z)*ω))/(20000000*sqrt(3)) + a4/tau
    du9 = a5/tau
   

    return(du1,du2,du3,du4,du5,du6,du7,du8,du9)
  end 
  

  function calc_tau(equations::PerturbationMomentSystem1D)
    @unpack tau = equations

    return(tau)

  end
  

  function shocktube(x, equations::PerturbationMomentSystem1D)
    rho_r = 1
    if (x[1] < 0)
      drho = 3 - rho_r
    else
      drho = 1 - rho_r
    end
  
    return drho
  end

  function initial_condition_constant(x, t, equations::PerturbationMomentSystem1D)
 
    return SVector(init_w0(x, t, equations::PerturbationMomentSystem1D), 
                   init_w0x(x, t, equations::PerturbationMomentSystem1D), 
                   init_w0y(x, t, equations::PerturbationMomentSystem1D), 
                   init_w1(x, t, equations::PerturbationMomentSystem1D),
                   init_w0xx(x, t, equations::PerturbationMomentSystem1D),
                   init_w0yy(x, t, equations::PerturbationMomentSystem1D),
                   init_w0xy(x, t, equations::PerturbationMomentSystem1D), 
                   init_w1x(x, t, equations::PerturbationMomentSystem1D),
                   init_w1y(x, t, equations::PerturbationMomentSystem1D))
  end
  
  
  @inline function init_w0(x, t, equations::PerturbationMomentSystem1D)
   
    rho_r = 1
    drho = shocktube(x, equations)
  
   
    w0 = 1 + drho / rho_r
    
   return w0
  end
  
  
  @inline function init_w0x(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack theta_r = equations 

    rho_r = 1
    drho = shocktube(x, equations)

    dv_x = 0
    
    w0x = dv_x / sqrt(theta_r) + (drho * dv_x)/(rho_r * sqrt(theta_r))

   return w0x
  end
  
  @inline function init_w0y(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack theta_r = equations 
    rho_r = 1
    drho = shocktube(x, equations)
    dv_y = 0
    
    w0y = dv_y / sqrt(theta_r) + (drho * dv_y)/(rho_r * sqrt(theta_r))
    
   return w0y
  end
  
  @inline function init_w1(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack theta_r = equations 
    rho_r = 1
    drho = shocktube(x, equations)
    dv_y = 0
    dv_x = 0
    dtheta = 0
    rho = 1 
    w1 = - (rho * (dv_x *dv_x + dv_y * dv_y) )/(3.0 * (rho_r * theta_r)) - (drho * dtheta)/(rho_r * theta_r) - dtheta / theta_r
    
   return w1
  end
  
  
  @inline function init_w0xx(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack theta_r = equations 

    rho_r = 1
    drho = shocktube(x, equations)
    dv_y = 0
    dv_x = 0

    sigma_xx = 0
    
    w0xx = 0.5 * sigma_xx/(rho_r * theta_r) + (2.0 * dv_x * dv_x - dv_y * dv_y)/(6.0 * theta_r) + (2.0 * drho * dv_x * dv_x - drho * dv_y * dv_y)/(6.0 * rho_r * theta_r)  
    
   return w0xx
  end
  
  @inline function init_w0yy(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack theta_r = equations 
    rho_r = 1
    drho = shocktube(x, equations)
    dv_y = 0
    dv_x = 0
    sigma_yy = 0
    
    w0yy = 0.5 * sigma_yy/(rho_r * theta_r) + (2.0 * dv_y * dv_y - dv_x * dv_x)/(6.0 * theta_r) + (2.0 * drho * dv_y * dv_y - drho * dv_x * dv_x)/(6.0 * rho_r * theta_r)
    
   return w0yy
  end
  
  
  @inline function init_w0xy(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack theta_r = equations 
    rho_r = 1
    drho = shocktube(x, equations)
    dv_y = 0
    dv_x = 0
    sigma_xy = 0
 
    
    w0xy = 0.5 * sigma_xy/(rho_r * theta_r) + 0.5 * ((rho_r + drho) * (dv_x * dv_y))/(rho_r * theta_r)
    
   return w0xy
  end
  
  
  
  @inline function init_w1x(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack theta_r = equations 
    rho_r = 1
    drho = shocktube(x, equations)
    dv_y = 0
    dv_x = 0
    sigma_xy = 0
    sigma_xx = 0
    q_x = 0

    dtheta = 0
    rho = 1

    w1x = - 2.0 * q_x / (5.0* rho_r * sqrt(theta_r).^3.0) - (2.0 * (sigma_xx * dv_x + sigma_xy * dv_y))/(5.0*rho_r* sqrt(theta_r).^3.0) - (dtheta * dv_x * rho)/ (rho_r * sqrt(theta_r).^3.0) - rho * (dv_x * dv_y^2 + dv_x^3)/(5.0 * rho_r * sqrt(theta_r).^3.0)
    
   return w1x
  end
  
  
  
  @inline function init_w1y(x, t, equations::PerturbationMomentSystem1D)
    
    @unpack theta_r = equations 
    dtheta = 0
    rho_r = 1
    drho = shocktube(x, equations)
    dv_y = 0
    dv_x = 0
    sigma_xy = 0
    sigma_yy = 0
    q_y = 0

    rho = 1
  
    w1y = - 2.0 * q_y / (5.0* rho_r * sqrt(theta_r).^3.0) - (2.0 * (sigma_xy * dv_x + sigma_yy * dv_y))/(5.0 * rho_r * sqrt(theta_r).^3.0) - (dtheta * dv_y * rho)/ (rho_r * sqrt(theta_r).^3.0) - rho * (dv_y * dv_x^2 + dv_y^3)/(5.0 * rho_r * sqrt(theta_r).^3.0) 
    
   return w1y
  end