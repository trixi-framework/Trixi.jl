module PointLocators


# Point structure
struct Point{NDIMS}
  x::NTuple{NDIMS, Float64}
end


# Main data structure for quadtree-like point locator
struct PointLocator{NDIMS}
  center::Point{NDIMS}
  length::Float64
  max_point_ids::Int
  point_ids::Vector{Int}
  children::Vector{PointLocator{NDIMS}}
end


function PointLocator{NDIMS}(center::Point, length::Float64, offset=0.0) where NDIMS
  # Use offset to avoid ambiguitites for points falling on coordinate lines
  center_ = Point(center.x .+ offset)
  length_ = length + 2 * offset

  # Use at most 20 points per locator node
  max_point_ids = 1
  point_ids = Vector{Int}()
  children = Vector{PointLocator{NDIMS}}()

  return PointLocator{NDIMS}(center_, length_, max_point_ids, point_ids, children)
end
function PointLocator{NDIMS}(center::Vector{Float64}, length::Float64, offset=0.0) where NDIMS
  PointLocator{NDIMS}(Point(Tuple(center)), length, offset)
end


# Return if locator has no children
is_leaf(pl::PointLocator) = isempty(pl.children)


# Insert point into locator
function insert!(pl::PointLocator, points::Vector{Point}, point::Point)
  # Check if locator is leaf
  if is_leaf(pl)
    # If locator is leaf, check if point already exists
    point_id = get_point_id(pl, points, point)

    # Return point_id if found
    if point_id > 0
      return point_id
    end

    # Otherwise check if there is enough room in current locator or if it needs to be refined
    if length(pl.point_ids) < pl.max_point_ids
      # Add point to this locator and to list of points
      point_id = length(points) + 1
      push!(points, point)
      push!(pl.point_ids, point_id)
      return point_id
    else
      # Refine locator
      refine!(pl)

      # Re-add to newly refined locator
      insert!(pl, points, point)
    end
  else
    # If locator is non-lef, find appropriate child locator and add point there
    child_locator = pl.children[get_child_id(pl, point)]
    insert!(child_locator, points, point)
  end
end
function insert!(pl::PointLocator, points::Vector{Point}, coordinates)
  insert!(pl, points, Point(coordinates))
end


# Refine point locator and move points to child locators
function refine!(pl::PointLocator{2})
  # Store for convenience
  dx = pl.length / 2
  x = pl.center.x[1]
  y = pl.center.x[2]

  # Add lower left child
  push!(pl.children, PointLocator{2}(Point((x - dx/2, y - dx/2)), dx))

  # Add lower right child
  push!(pl.children, PointLocator{2}(Point((x + dx/2, y - dx/2)), dx))

  # Add upper left child
  push!(pl.children, PointLocator{2}(Point((x - dx/2, y + dx/2)), dx))

  # Add upper right child
  push!(pl.children, PointLocator{2}(Point((x + dx/2, y + dx/2)), dx))
end
function refine!(pl::PointLocator{3})
  # Store for convenience
  dx = pl.length / 2
  x = pl.center.x[1]
  y = pl.center.x[2]
  z = pl.center.x[3]

  # Add bottom lower left child
  push!(pl.children, PointLocator{3}(Point((x - dx/2, y - dx/2, z - dx/2)), dx))

  # Add bottom lower right child
  push!(pl.children, PointLocator{3}(Point((x + dx/2, y - dx/2, z - dx/2)), dx))

  # Add bottom upper left child
  push!(pl.children, PointLocator{3}(Point((x - dx/2, y + dx/2, z - dx/2)), dx))

  # Add bottom upper right child
  push!(pl.children, PointLocator{3}(Point((x + dx/2, y + dx/2, z - dx/2)), dx))

  # Add top lower left child
  push!(pl.children, PointLocator{3}(Point((x - dx/2, y - dx/2, z + dx/2)), dx))

  # Add top lower right child
  push!(pl.children, PointLocator{3}(Point((x + dx/2, y - dx/2, z + dx/2)), dx))

  # Add top upper left child
  push!(pl.children, PointLocator{3}(Point((x - dx/2, y + dx/2, z + dx/2)), dx))

  # Add top upper right child
  push!(pl.children, PointLocator{3}(Point((x + dx/2, y + dx/2, z + dx/2)), dx))
end


# Get id of child locator for given point
function get_child_id(pl::PointLocator{2}, point::Point)
  if point.x[2] < pl.center.x[2]
    if point.x[1] < pl.center.x[1]
      # Lower left child
      return 1
    else
      # Lower right child
      return 2
    end
  else
    if point.x[1] < pl.center.x[1]
      # Upper left child
      return 3
    else
      # Upper right child
      return 4
    end
  end
end
function get_child_id(pl::PointLocator{3}, point::Point)
  if point.x[3] < pl.center.x[3]
    if point.x[2] < pl.center.x[2]
      if point.x[1] < pl.center.x[1]
        # Bottom lower left child
        return 1
      else
        # Bottom lower right child
        return 2
      end
    else
      if point.x[1] < pl.center.x[1]
        # Bottom upper left child
        return 3
      else
        # Bottom upper right child
        return 4
      end
    end
  else
    if point.x[2] < pl.center.x[2]
      if point.x[1] < pl.center.x[1]
        # Top lower left child
        return 5
      else
        # Top lower right child
        return 6
      end
    else
      if point.x[1] < pl.center.x[1]
        # Top upper left child
        return 7
      else
        # Top upper right child
        return 8
      end
    end
  end
end


# Get point id if point exists or zero otherwise
function get_point_id(pl::PointLocator, points::Vector{Point}, point::Point)
  # Iterate over point ids, extract point coordinates and compare to point
  for point_id in pl.point_ids
    if all(isapprox.(point.x, points[point_id].x, atol=1e-13))
      return point_id
    end
  end

  # If no point was found, return zero
  return 0
end


end # module PointLocators
