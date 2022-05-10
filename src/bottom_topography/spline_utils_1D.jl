# Sorting the inputs
function sort_data(x::Vector{Float64},y::Vector{Float64})
      
    orig_data = hcat(x,y)
    sort_data = orig_data[sortperm(orig_data[:,1]), :]
    
    u = sort_data[:,1]
    t = sort_data[:,2]
    
    return u,t
  end