module Containers

export AbstractContainer

# Abstract base type - all containers that want to use these features must inherit from it
abstract type AbstractContainer end

# Add generic functions for which concrete containers must add implementations
function invalidate! end
function raw_copy! end
function move_connectivity! end
function delete_connectivity! end


# Auxiliary copy function to copy data between containers
function copy_data!(target::AbstractArray, source::AbstractArray,
                    first::Int, last::Int, destination::Int, block_size::Int=1)
  count = last - first + 1
  if destination <= first || destination > last
    # In this case it is safe to copy forward (left-to-right) without overwriting data
    for i in 0:(count-1), j in 1:block_size
      target[block_size*(destination+i-1) + j] = source[block_size*(first+i-1) + j]
    end
  else
    # In this case we need to copy backward (right-to-left) to prevent overwriting data
    for i in reverse(0:(count-1)), j in 1:block_size
      target[block_size*(destination+i-1) + j] = source[block_size*(first+i-1) + j]
    end
  end
end


# Inquire about capacity and size
capacity(c::AbstractContainer) = c.capacity
Base.length(c::AbstractContainer) = c.length
size(c::AbstractContainer) = (length(c),)


# Increase container length by `count` elements
function append!(c::AbstractContainer, count::Int)
  @assert count >= 0 "Count must be non-negative"
  @assert count + length(c) <= capacity(c) "New length would exceed capacity"

  # First, invalidate range (to be sure that no sensible values are accidentally left there)
  invalidate!(c, length(c) + 1, length(c) + count)

  # Then, increase container length
  c.length += count
end
append!(c::AbstractContainer) = append(c, 1)


# Decrease container length by `count` elements
function shrink!(c::AbstractContainer, count::Int)
  @assert count >= 0
  @assert length(c) >= count

  # Rely on remove&shift to do The Right Thing
  remove_shift!(c, length(c) - count + 1, length())
end
shrink!(c::AbstractContainer) = shrink(c, 1)


# Copy data range from source to target container.
#
# Calls `raw_copy` internally, which must be implemented for each concrete type
# inheriting from AbstractContainer.
function copy!(target::AbstractContainer, source::AbstractContainer,
               first::Int, last::Int, destination::Int)
  @assert 1 <= first <= length(source) "First cell out of range"
  @assert 1 <= last <= length(source) "Last cell out of range"
  @assert 1 <= destination <= length(target) "Destination out of range"
  @assert destination + (last - first) <= length(target) "Target range out of bounds"

  # Return if copy would be a no-op
  if last < first || (source === target && first == destination)
    return
  end

  raw_copy!(target, source, first, last, destination)
end


# Convenience method to copy a single element
function copy!(target::AbstractContainer, source::AbstractContainer, from::Int, destination::Int)
  copy!(target, source, from, from, destination)
end


# Convenience method for copies within a single container
function copy!(c::AbstractContainer, first::Int, last::Int, destination::Int)
  copy!(c, c, first, last, destination)
end


# Convenience method for copying a single element within a single container
function copy!(c::AbstractContainer, from::Int, destination::Int)
  copy!(c, c, from, from, destination)
end


# Move elements in a way that preserves connectivity.
function move!(c::AbstractContainer, first::Int, last::Int, destination::Int)
  @assert 1 <= first <= length(c) "First cell $first out of range"
  @assert 1 <= last <= length(c) "Last cell $last out of range"
  @assert 1 <= destination <= length(c) "Destination $destination out of range"
  @assert destination + (last - first) <= length(c) "Target range out of bounds"

  # Return if move would be a no-op
  if last < first || first == destination
    return
  end

  # Copy cells to new location
  raw_copy!(c, first, last, destination)

  # Move connectivity
  move_connectivity!(c, first, last, destination)


  # Invalidate original cell locations (unless they already contain new data due to overlap)
  # 1) If end of desination range is within original range, shift first_invalid to the right
  count = last - first + 1
  first_invalid = (first <= destination + count - 1 <= last) ? destination + count : first
  # 2) If beginning of destination range is within original range, shift last_invalid to the left
  last_invalid = (first <= destination <= last) ? destination - 1 : last
  # 3) Invalidate range
  invalidate!(c, first_invalid, last_invalid)
end
move!(c::AbstractContainer, from::Int, destination::Int) = move!(c, from, from, destination)


# Swap two elements in a container while preserving element connectivity.
function swap!(c::AbstractContainer, a::Int, b::Int)
  @assert 1 <= a <= length(c) "a out of range"
  @assert 1 <= b <= length(c) "b out of range"

  # Return if swap would be a no-op
  if a == b
    return
  end

  # Move a to dummy location
  raw_copy!(c, a, c.dummy)
  move_connectivity(c, a, c.dummy)

  # Move b to a
  raw_copy!(c, b, a)
  move_connectivity(c, b, a)

  # Move from dummy location to b
  raw_copy!(c, c.dummy, b)
  move_connectivity(c, c.dummy, b)

  # Invalidate dummy to be sure
  invalidate(c, c.dummy)
end


# Insert blank elements in container, shifting the following elements back.
#
# After a call to insert!, the range `position:position + count - 1` will be available for use.
function insert!(c::AbstractContainer, position::Int, count::Int)
  @assert 1 <= position <= length(c) + 1 "Insert position out of range"
  @assert count >= 0 "Count must be non-negative"
  @assert count + length(c) <= capacity(c) "New length would exceed capacity"

  # Return if insertation would be a no-op
  if count == 0
    return
  end

  # Append and return if insertion is beyond last current element
  if position == length(c) + 1
    append!(c, count)
    return
  end

  # Increase length
  c.length += count

  # Move original cells that currently occupy the insertion region, unless
  # insert position is one beyond previous length
  if position <= length(c) - count
    move!(c, position, length(c) - count, position + count)
  end
end
insert!(c) = insert!(c, position, 1)


# Erase elements from container, deleting their connectivity and then invalidating their data.
function erase!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= length(c) "First cell out of range"
  @assert 1 <= last <= length(c) "Last cell out of range"

  # Return if eraseure would be a no-op
  if last < first
    return
  end

  # Delete connectivity and invalidate cells
  delete_connectivity!(c, first, last)
  invalidate!(c, first, last)
end
erase!(c::AbstractContainer, id::Int) = erase!(c, id, id)


# Remove cells and shift existing cells forward to close the gap
function remove_shift!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= length(c) "First cell out of range"
  @assert 1 <= last <= length(c) "Last cell out of range"

  # Return if removal would be a no-op
  if last < first
    return
  end

  # Delete connectivity of cells to be removed
  delete_connectivity!(c, first, last)

  if last == length(c)
    # If everything up to the last cell is removed, no shifting is required
    invalidate!(c, first, last)
  else
    # Otherwise, the corresponding cells are moved forward
    move!(c, last + 1, length(c), first)
  end

  # Reduce length
  count = last - first + 1
  c.length -= count
end
remove_shift!(c::AbstractContainer, id::Int) = remove_shift!(c, id, id)


# Remove cells and fill gap with cells from the end of the container (to reduce copy operations)
function remove_fill!(c::AbstractContainer, first::Int, last::Int)
  @assert 1 <= first <= length(c) "First cell out of range"
  @assert 1 <= last <= length(c) "Last cell out of range"

  # Return if removal would be a no-op
  if last < first
    return
  end

  # Delete connectivity of cells to be removed and then invalidate them
  delete_connectivity!(c, first, last)
  invalidate!(c, first, last)

  # Copy cells from end (unless last is already the last cell)
  count = last - first + 1
  if last < length(c)
    move(c, max(length(c) - count, last + 1), length(c), first)
  end

  # Reduce length
  c.length -= count
end


# Reset container to zero-length and with a new capacity
function reset!(c::AbstractContainer, capacity::Int)
  @assert capacity >=0

  c.capacity = capacity
  c.length = 0
  c.dummy = capacity + 1
  reset_data_structures!(c::AbstractContainer)
end


# Invalidate all elements and set length to zero.
function clear!(c::AbstractContainer)
  invalidate!(c)
  c.length = 0
end

end # module Containers
