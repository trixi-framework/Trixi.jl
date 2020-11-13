using P4est
using CBinding

@cstruct quad_inner_data_t {
    id::Int64;
    oldids::Int64[4];
    # oldid2::Int64;
    # oldid3::Int64;
    # oldid4::Int64;
}

function volumeIterate(info::Ptr{P4est.p4est_iter_volume_info_t} , user_data::Ptr{Cvoid})
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


function faceIterate(info::Ptr{P4est.p4est_iter_face_info_t}, user_data_ptr::Ptr{Cvoid})
 
      p4est = info.p4est

      local_num_quads = Int64(p4est.local_num_quadrants)
  
      Conn = unsafe_wrap(Array, Ptr{Int32}(user_data_ptr), (11,local_num_quads); own = false)
      
  
      sides = [unsafe_wrap(P4est.p4est_iter_face_side_t, info.sides.array),
          unsafe_wrap(P4est.p4est_iter_face_side_t, info.sides.array + info.sides.elem_size)]
    
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

function setOldIdtoZero(info::Ptr{P4est.p4est_iter_volume_info_t} , user_data::Ptr{Cvoid})
 p4est = info.p4est
 quad = info.quad
 quadid = info.quadid
#  @show quad.p.user_data
#  dataptr = Ptr{quad_inner_data_t}(quad.p.user_data)
 dataptr = unsafe_wrap(quad_inner_data_t, quad.p.user_data)
 dataptr.id = quadid + 1
 dataptr.oldids = zeros(4) #2 * Ndims
 dataptr.oldids[1] =  dataptr.id
 return nothing
end


# static int
# refine_fn(p4est_t *p4est, p4est_topidx_t which_tree,
#           p4est_quadrant_t *q) {
#     int *ToRefine = (int *) p4est->user_pointer;

#     p4est_inner_data_t *dataquad = (p4est_inner_data_t *) q->p.user_data;
#    
#     if ((ToRefine[dataquad->ElementID - 1] > 0 && ToRefine[dataquad->ElementID - 1] > q->level)
#         || (-ToRefine[dataquad->ElementID - 1] - 1 > q->level)) {
#         // printf("REFINE!!!!!!!!!!!!!!! \n");
#         return 1;
#     } else {
#         return 0;
#     }
# }

function refine_function(p4est::Ptr{P4est.p4est_t},
                        which_tree::P4est.p4est_topidx_t,
                        q::Ptr{P4est.p4est_quadrant_t})
    user_data_ptr = p4est.user_pointer
    # local_num_quads = Int64(p4est.local_num_quadrants)
    # quadinfo = unsafe_wrap(Array, Ptr{Int32}(user_data), (4,local_num_quads); own = false)
    dataptr = unsafe_wrap(quad_inner_data_t, q.p.user_data)
    
    to_refine = unsafe_wrap(Array, Ptr{Int32}(user_data_ptr), (dataptr.id); own = false)
    elem_id = dataptr.id
    # @show dataptr.id, to_refine[dataptr.id]
    if (to_refine[elem_id] > 0) #  && to_refine[elem_id] > q.level
        # @show 1, "REFINE"
        # return Cint(0) #TODO: REMOVE
        return Cint(1)
    else
        return Cint(0)
    end
end

function coarse_function(p4est::Ptr{P4est.p4est_t},
                        which_tree::P4est.p4est_topidx_t,
                        children_array_ptr::Ptr{Ptr{P4est.p4est_quadrant_t}})

    user_data_ptr = p4est.user_pointer
    children = unsafe_wrap(Array, Ptr{Ptr{P4est.p4est_quadrant_t}}(children_array_ptr), 4; own = false)
    # @show children[1].x
    # @show children[2].y
    # @show children[3].y
    # @show children[4].x
    # elem_id = 0   #children[1].quadid
    Coarse4 = 0; 
    
    for i = 1:4 #2*NDIMS
        data = unsafe_wrap(quad_inner_data_t, children[i].p.user_data)
 
        # @show data.id
        if (data.oldids[1] < 0 || data.id == 0)
            # @show "Not TO Coarse"
            return Cint(0)# This Element was refined and we don't need to check it.
        end
        to_coarse = unsafe_wrap(Array, Ptr{Int32}(user_data_ptr), (data.id); own = false)
        if (to_coarse[data.id] < 0)
        #    if (children[i].level > (- to_corase[data.id] - 1)) 
                # The level compared with -1*to_coarse[] - 1
                Coarse4 += 1
        #    else
        #        @show "Not TO Coarse"
        #        return Cint(0)
        #    end
        end
    end
    if (Coarse4 == 4) 
        # @show "TO Coarse"
        return Cint(0)  #TODO: return Cint(1)
    else
        # @show " 1 Not TO Coarse"
        return Cint(0)
    end
end



function replace_quads(p4est::Ptr{P4est.p4est_t}, #p4est_t * p4est
        which_tree::P4est.p4est_topidx_t, #p4est_topidx_t which_tree,
        num_outgoing::Int32, #int num_outgoing,
        outgoing_array_ptr::Ptr{Ptr{P4est.p4est_quadrant_t}}, #p4est_quadrant_t * outgoing[],
        num_incoming::Int32,#int num_incoming
        incoming_array_ptr::Ptr{Ptr{P4est.p4est_quadrant_t}}#p4est_quadrant_t * incoming[]
        )
        # @show num_outgoing
        # @show num_incoming

        incoming = unsafe_wrap(Array, Ptr{Ptr{P4est.p4est_quadrant_t}}(incoming_array_ptr), num_incoming; own = false)
        outgoing = unsafe_wrap(Array, Ptr{Ptr{P4est.p4est_quadrant_t}}(outgoing_array_ptr), num_outgoing; own = false)
        if (num_outgoing > 1) # * Coarsening
            @assert num_incoming == 1
            parentquaddata = unsafe_wrap(quad_inner_data_t, incoming[1].p.user_data)
            for i = 1:4
                childquaddata = unsafe_wrap(quad_inner_data_t, outgoing[i].p.user_data);
                parentquaddata.oldids[i] = childquaddata[i].id
            end
            # for (i = 0; i < P8EST_CHILDREN; i++) {
            #     childquaddata = (p4est_inner_data_t *) outgoing[i]->p.user_data;
              
            #     parentquaddata->OldElementID[i] = childquaddata->OldElementID[0];
    
            # }
        # * 

        else # * NOTE: Refine
            @assert num_outgoing == 1
            @assert num_incoming > 1
            parentquaddata = unsafe_wrap(quad_inner_data_t, outgoing[1].p.user_data)
            
            if (parentquaddata.oldids[2] > 0)
                # * This quad was Coarsed but Refined again by balance 2:1
                for i = 1:4
                    childquaddata = unsafe_wrap(quad_inner_data_t, incoming[i].p.user_data);
                    childquaddata.oldids[1] = parentquaddata.oldids[i]
                    for j = 2:4
                        childquaddata.oldids[j] = 0
                    end
                end
                
                #     for (i = 0; i < P8EST_CHILDREN; i++) {
            #         childquaddata = (p4est_inner_data_t *) incoming[i]->p.user_data;
            #         childquaddata->OldElementID[0] = parentquaddata->OldElementID[i];
            
            #         int i1;
            #         for (i1 = 1; i1 < P8EST_CHILDREN; i1++) {
                #             childquaddata->OldElementID[i1] = 0;
                #         }
                # }
            else
                for i = 1:4
                    childquaddata = unsafe_wrap(quad_inner_data_t, incoming[i].p.user_data);
                    childquaddata.oldids[1] = - parentquaddata.oldids[1]
                    for j = 2:4
                        childquaddata.oldids[j] = 0
                    end
                    childquaddata.id = 0
            # } else {
            #     for (i = 0; i < P8EST_CHILDREN; i++) {
            #         childquaddata = (p4est_inner_data_t *) incoming[i]->p.user_data;
            #         childquaddata->OldElementID[0] = -parentquaddata->OldElementID[0];
            #         int j;
            #         for (j = 1; j < P8EST_CHILDREN; j++) {
            #             childquaddata->OldElementID[j] = 0;
            #         }
            #         childquaddata->ElementID = 0;
            #     }
            # }

                end
            end
        end
        return nothing
end #function

function GetChanges(info::Ptr{P4est.p4est_iter_volume_info_t} , user_data::Ptr{Cvoid})
    p4est = info.p4est
    local_num_quads = Int64(p4est.local_num_quadrants)
    quad = info.quad
    quadid = info.quadid + 1
    dataptr = unsafe_wrap(quad_inner_data_t, quad.p.user_data)
    # dataptr.id = quadid + 1
    Changes = unsafe_wrap(Array, Ptr{Int64}(user_data), (4,local_num_quads); own = false)
    for i = 1:4
        Changes[i, quadid] = dataptr.oldids[i]
    end
    return nothing
end