module PointLocators


# Point structure
Point = NamedTuple{(:x, :y), Tuple{Float64, Float64}}


# Main data structure for quadtree-like point locator
struct PointLocator
  center::Point
  length::Float64
  max_point_ids::Int
  point_ids::Vector{Int}
  children::Vector{PointLocator}
end


function PointLocator(center::Point, length::Float64, offset=0.0)
  # Use offset to avoid ambiguitites for points falling on coordinate lines
  center_ = Point((center.x + offset, center.y + offset))
  length_ = length + 2 * offset

  # Use at most 20 points per locator node
  max_point_ids = 20
  point_ids = Vector{Int}()
  children = Vector{PointLocator}()

  return PointLocator(center_, length_, max_point_ids, point_ids, children)
end
function PointLocator(center::Vector{Float64}, length::Float64, offset=0.0)
  PointLocator(Point((center[1], center[2])), length, offset)
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
function insert!(pl::PointLocator, points::Vector{Point}, x::Float64, y::Float64)
  insert!(pl, points, Point((x, y)))
end


# Refine point locator and move points to child locators
function refine!(pl::PointLocator)
  # Store for convenience
  dx = pl.length / 2
  x = pl.center.x
  y = pl.center.y

  # Add lower left child
  push!(pl.children, PointLocator(Point((x - dx/2, y - dx/2)), dx))

  # Add lower right child
  push!(pl.children, PointLocator(Point((x + dx/2, y - dx/2)), dx))

  # Add upper left child
  push!(pl.children, PointLocator(Point((x - dx/2, y + dx/2)), dx))

  # Add upper right child
  push!(pl.children, PointLocator(Point((x + dx/2, y + dx/2)), dx))
end


# Get id of child locator for given point
function get_child_id(pl::PointLocator, point::Point)
  if point.y < pl.center.y
    if point.x < pl.center.x
      # Lower left child
      return 1
    else
      # Lower right child
      return 2
    end
  else
    if point.x < pl.center.x
      # Upper left child
      return 3
    else
      # Upper right child
      return 4
    end
  end
end


# Get point id if point exists or zero otherwise
function get_point_id(pl::PointLocator, points::Vector{Point}, point::Point)
  # Iterate over point ids, extract point coordinates and compare to point
  for point_id in pl.point_ids
    x = points[point_id].x
    y = points[point_id].y
    if isapprox(point.x, x, atol=1e-13) && isapprox(point.y, y, atol=1e-13)
      return point_id
    end
  end

  # If no point was found, return zero
  return 0
end


end # module PointLocators
