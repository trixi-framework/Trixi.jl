# using CBinding



function volumeIterate(info::Ptr{P4est.p4est_iter_volume_info_t} , user_data::Ptr{Cvoid})
    # info = unsafe_wrap(p4est_iter_volume_info_t, info_ptr)
    # @show info.quadid
    # quad = unsafe_wrap(p4est_quadrant_t, info.quad)
    # @show quad.level
    # println("11") 
    p4est = info.p4est
    local_num_quads = Int64(p4est.local_num_quadrants)
    quadinfo = unsafe_wrap(Array, Ptr{Int32}(user_data), (4,local_num_quads); own = false)
    quadid = info.quadid
    quad = info.quad
    quadinfo[1, quadid + 1] = quadid + 1
    quadinfo[2, quadid + 1] = quad.level
    quadinfo[3, quadid + 1] = trunc(Int, 1024 * quad.x / 2147483647) 
    quadinfo[4, quadid + 1] = trunc(Int, 1024 * quad.y / 2147483647) 
    return nothing
end

CvolumeIterate = @cfunction(volumeIterate, Cvoid, (Ptr{P4est.p4est_iter_volume_info_t}, Ptr{Cvoid}))

function faceIterate(info::Ptr{P4est.p4est_iter_face_info_t}, user_data_ptr::Ptr{Cvoid})
      # info = unsafe_wrap(P4est.p4est_iter_face_info_t, info_ptr)
      p4est = info.p4est
      # @show p4est
      # @show info_ptr.sides.elem_size
      # info = unsafe_wrap(P4est.p4est_iter_face_info_t, info_ptr)
      # p4est = unsafe_wrap(P4est.p4est_t, info.p4est)
      local_num_quads = Int64(p4est.local_num_quadrants)
      # # unsafe_wrap(Array, pointer::Ptr{T}, dims; own = false)
      # # @show local_num_quads
      Conn = unsafe_wrap(Array, Ptr{Int32}(user_data_ptr), (11,local_num_quads); own = false)
      
  
      sides = [unsafe_wrap(P4est.p4est_iter_face_side_t, info.sides.array),
          unsafe_wrap(P4est.p4est_iter_face_side_t, info.sides.array + info.sides.elem_size)]
      # @show sides[1].is.full.quad
      # return
      if (sides[1].is_hanging == 0 && sides[2].is_hanging == 0)
  
          quads = [sides[1].is.full.quad, sides[2].is.full.quad]
          quadIds = [sides[1].is.full.quadid, sides[2].is.full.quadid]
          Conn[1, quadIds[1] + 1] = quads[1].level
          Conn[1, quadIds[2] + 1] = quads[2].level
  
          Conn[2, quadIds[1] + 1] = trunc(Int, 1024 * quads[1].x / 2147483647)
          Conn[2, quadIds[2] + 1] = trunc(Int, 1024 * quads[2].x / 2147483647)
  
          Conn[3, quadIds[1] + 1] = trunc(Int, 1024 * quads[1].y / 2147483647)
          Conn[3, quadIds[2] + 1] = trunc(Int, 1024* quads[2].y / 2147483647)
  
          Conn[sides[1].face * 2 + 4,quadIds[1] + 1] = quadIds[2] + 1
          Conn[sides[2].face * 2 + 4,quadIds[2] + 1] = quadIds[1] + 1
      else
          BigSide = 2
          HangSide = 1
          if sides[2].is_hanging == 1
              BigSide = 1
              HangSide = 2
          end
  
          quadBig = sides[BigSide].is.full.quad
          quadBigId = sides[BigSide].is.full.quadid
          quadsHanging = [sides[HangSide].is.hanging.quad[1],
                          sides[HangSide].is.hanging.quad[2]]
          quadHangIds = [sides[HangSide].is.hanging.quadid[1], sides[HangSide].is.hanging.quadid[2]]
        #   @show quadHangIds
          Conn[1, quadBigId + 1] = quadBig.level
          Conn[1, quadHangIds[1] + 1] = quadsHanging[1].level
          Conn[1, quadHangIds[2] + 1] = quadsHanging[2].level
  
          Conn[2, quadBigId + 1] = trunc(Int, 1024 * quadBig.x / 2147483647)
          Conn[2, quadHangIds[1] + 1] = trunc(Int, 1024 * quadsHanging[1].x / 2147483647)
          Conn[2, quadHangIds[2] + 1] = trunc(Int, 1024 * quadsHanging[2].x / 2147483647)
  
          Conn[3, quadBigId + 1] = trunc(Int, 1024 * quadBig.y / 2147483647)
          Conn[3, quadHangIds[1] + 1] = trunc(Int, 1024 * quadsHanging[1].y / 2147483647)
          Conn[3, quadHangIds[2] + 1] = trunc(Int, 1024 * quadsHanging[2].y / 2147483647)
  
          Conn[sides[BigSide].face * 2 + 4,quadBigId + 1] = quadHangIds[1] + 1
          Conn[sides[BigSide].face * 2 + 5,quadBigId + 1] = quadHangIds[2] + 1
  
          Conn[sides[HangSide].face * 2 + 4,quadHangIds[1] + 1] = quadBigId + 1
          Conn[sides[HangSide].face * 2 + 5,quadHangIds[1] + 1] = quadBigId + 1
  
          Conn[sides[HangSide].face * 2 + 4,quadHangIds[2] + 1] = quadBigId + 1
          Conn[sides[HangSide].face * 2 + 5,quadHangIds[2] + 1] = quadBigId + 1
  
      end
  
      return nothing
end

CfaceIterate = @cfunction(faceIterate, Cvoid, (Ptr{P4est.p4est_iter_face_info_t}, Ptr{Cvoid}))
