using Distributions: Uniform

function trainfunction(func,x,y)
    if func == 1
        a1 = 9  
        b1 = -4 
        u = a1*x + b1*y 
    elseif func == 2
        a1 = 2  
        b1 = -1 
        u = a1*x + b1*y
    elseif func == 3
        a1 = 0.5  
        b1 = -3 
        u = a1*x + b1*y  
    elseif func == 4
        a1 = -2 
        b1 = 9 
        u = a1*x + b1*y 
    elseif func == 5
        a = 0.9
        b = 0.2
        u = a*sin(pi*x)+b*cos(pi*y)
    elseif func == 6
        a = 0.8
        b = -0.3
        u = a*sin(2*pi*x)+b*sin(2*pi*y)
    elseif func == 7
        a = 0.1
        b = 0.6
        u = a*sin(pi*x)+b*sin(pi*y)
    elseif func == 8
        a = [-0.4]    
        b = [-0.7] 
        u = a[1]*sin(pi*x)+b[1]cos(pi*y)
    elseif func == 9
        a = [-0.9, 0.1]#rand(Uniform(-1,1),3)
        b = [0.4, 0.2]#rand(Uniform(-1,1),3)
        u1 = a[1]*sin(pi*x)+b[1]cos(pi*y)
        u = u1 + a[2]*sin(2*pi*x)+b[2]cos(2*pi*y)
    elseif func == 10
        a = [-1, 1.0]#rand(Uniform(-1,1),3)
        b = [-0.2, 0.5]#rand(Uniform(-1,1),3)
        u1 = a[1]*sin(pi*x)+b[1]cos(pi*y)
        u = u1 + a[2]*sin(2*pi*x)+b[2]cos(2*pi*y)
    elseif func == 11
        a = [0.9, -0.6, -0.3]#rand(Uniform(-1,1),3)
        b = [1.0, -0.1, -0.7]#rand(Uniform(-1,1),3)
        u1 = a[1]*sin(pi*x)+b[1]cos(pi*y)
        u2 = u1 + a[2]*sin(2*pi*x)+b[2]cos(2*pi*y)
        u = u2 + a[3]*sin(3*pi*x)+b[3]cos(3*pi*y)
    elseif func == 12
        a = [-0.2, 0.6, -0.8]#rand(Uniform(-1,1),3)
        b = [-1.0, -0.1, 0.7]#rand(Uniform(-1,1),3)
        u1 = a[1]*sin(pi*x)+b[1]cos(pi*y)
        u2 = u1 + a[2]*sin(2*pi*x)+b[2]cos(2*pi*y)
        u = u2 + a[3]*sin(3*pi*x)+b[3]cos(3*pi*y)
    elseif func == 13
        a =[0.9, 0.1, -0.4, -0.1, 0.3, -0.2]    # rand(Uniform(-1,1),6)
        u = exp(a[1]*((x-a[2])^2+(y-a[3])^2)) + exp(a[4]*((x-a[5])^2+(y-a[6])^2))
    elseif func == 14
        a =[0.3, -0.1, -0.4, 0.8, 0.3, -0.2]    # rand(Uniform(-1,1),6)
        u = exp(a[1]*((x-a[2])^2+(y-a[3])^2)) + exp(a[4]*((x-a[5])^2+(y-a[6])^2))
    elseif func == 15
        a =[-0.8, 0.4, -0.1, -0.3, 0.7, 0.9]    # rand(Uniform(-1,1),6)
        u = exp(a[1]*((x-a[2])^2+(y-a[3])^2)) + exp(a[4]*((x-a[5])^2+(y-a[6])^2))
    elseif func == 16
        a =[1.0, 0.1, -0.1, -0.6, 0.3, -0.9]    # rand(Uniform(-1,1),6)
        u = exp(a[1]*((x-a[2])^2+(y-a[3])^2)) + exp(a[4]*((x-a[5])^2+(y-a[6])^2))
    elseif func == 17
        a =[-0.2, 0.6, -0.1, -0.4, 0.9, -0.8]   # rand(Uniform(-1,1),6)
        u = exp(a[1]*((x-a[2])^2+(y-a[3])^2)) + exp(a[4]*((x-a[5])^2+(y-a[6])^2))
    elseif func == 18
        a = 45      #rand(Uniform(-100, 100)) 
        m = -0.2    #rand(Uniform(-1,1)) 
        x0 = 0.4    #rand(Uniform(-0.5, 0.5)) 
        y0 = -0.3   #rand(Uniform(-0.5, 0.5)) 
        u =  a*abs((y-y0)-m*(x-x0))
    elseif func == 19
        ui = [0.9, -0.6, 0.3, -0.1] #rand(Uniform(-1,1),4)
        m = 5       #rand(Uniform(0,20))
        x0 = 0.5    #rand(Uniform(-0.5, 0.5))
        y0 = -0.5   #rand(Uniform(-0.5, 0.5))
        h1 = y0+m*(x-x0)
        h2 = y0-(1/m)*(x-x0)
        if y <= h1 && y < h2
            u=ui[1]
        elseif y >= h1 && y > h2
            u=ui[2]
        elseif y < h1 && y >= h2
            u=ui[3]
        elseif y > h1 && y <= h2
            u=ui[4]
        elseif y == h1 && y == h2
            u=ui[1]
        end
    end
    return u
end

function troubledcellfunctionabs(x, y, a, m, x0, y0)
    u =  a*abs((y-y0)-m*(x-x0))
    return u
end

function troubledcellfunctionstep(x, y, ui, m, x0, y0)
    h1 = y0+m*(x-x0)
    h2 = y0-(1/m)*(x-x0)

    if y <= h1 && y < h2
        u=ui[1]
    elseif y >= h1 && y > h2
        u=ui[2]
    elseif y < h1 && y >= h2
        u=ui[3]
    elseif y > h1 && y <= h2
        u=ui[4]
    elseif y == h1 && y == h2
        u=ui[1]
    end
    return u
end

function good_cell(node_coord, length, func, m, x0, y0)
    x = node_coord[1] + length/2
    y = node_coord[2] + length/2
    if func == 1
        if -(3/2)*length < (y-y0)-m*(x-x0) && (y-y0)-m*(x-x0) <(3/2)*length #(y-y0)-m*(x-x0) in [-3/2*length,3/2*length]
            return false
        else 
            return true
        end
    elseif func == 2
        if (-(3/2)*length < y-y0-m*(x-x0) && y-y0-m*(x-x0) < (3/2)*length) || (-(3/2)*length < y-y0+(1/m)*(x-x0) && y-y0+(1/m)*(x-x0) < (3/2)*length) 
            #y-y0-m*(x-x0) in [-3/2*length,3/2*length] || y-y0+(1/m)*(x-x0) in [-3/2*length,3/2*length]
            return false
        else
            return true
        end
    end
end

function validfunction(func,x,y)
    if func == 1
        a = [-0.2, 0.8, -0.7]#rand(Uniform(-1,1),3)
        b = [-1.0, 0.4, -0.1]#rand(Uniform(-1,1),3)
        u1 = a[1]*sin(pi*x)+b[1]cos(pi*y)
        u2 = u1 + a[2]*sin(2*pi*x)+b[2]cos(2*pi*y)
        u = u2 + a[3]*sin(3*pi*x)+b[3]cos(3*pi*y)
    elseif func == 2
        a = [0.4, -0.8, 0.1]#rand(Uniform(-1,1),3)
        b = [0.1, -0.6, 0.9]#rand(Uniform(-1,1),3)
        u1 = a[1]*sin(pi*x)+b[1]cos(pi*y)
        u2 = u1 + a[2]*sin(2*pi*x)+b[2]cos(2*pi*y)
        u = u2 + a[3]*sin(3*pi*x)+b[3]cos(3*pi*y)    
    elseif func == 3
        a = [0.4]
        b = [0.1]
        u = a[1]*sin(pi*x)+b[1]cos(pi*y) 
    elseif func == 4
        a = [-0.4]
        b = [0.9]
        u = a[1]*sin(pi*x)+b[1]sin(pi*y)
    elseif func == 5
        a = [0.4]
        b = [-0.7]
        u = a[1]*sin(2*pi*x)+b[1]sin(2*pi*y)        
    elseif func == 6
        a =[0.3, -0.1]# rand(Uniform(-1,1),6)
        u = a[1]*x + a[2]*x^2 
    elseif func == 7
        a =[-0.1, 0.6, 0.8]# rand(Uniform(-1,1),6)
        u = a[1]*x + a[2]*x^2 + a[3]*x^3
    elseif func == 8
        a =[-0.5, -0.1, -0.8, 0.8]# rand(Uniform(-1,1),6)
        u = a[1]*x + a[2]*x^2 + a[3]*x^3 + a[4]*x^4
    elseif func == 9
        a =[0.8, -0.4, 0.1]# rand(Uniform(-1,1),6)
        u = a[1]*x + a[2]*x^2 + a[3]*x^3
    end
    return u
end






