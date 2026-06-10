module TrixiFastBroadcastExt

using FastBroadcast: FastBroadcast, Serial, Threaded
using Trixi: TrixiStateVector

# Opt TrixiStateVector out of FastBroadcast's threaded broadcast path.
#
# fast_materialize_threaded! chunks the destination into views and broadcasts
# the full-size source onto each smaller chunk, producing DimensionMismatch.
# Routing through Serial instead reaches:
#   fast_materialize!(Serial(), dst, bc)  →  materialize!(dst, bc)  →  copyto!(dst, bc)
# which dispatches to TrixiStateVector's custom copyto! (which uses @threaded internally).
@inline function FastBroadcast.fast_materialize!(::Threaded, dst::TrixiStateVector,
                                                 bc::Base.Broadcast.Broadcasted)
    return FastBroadcast.fast_materialize!(Serial(), dst, bc)
end

end # module TrixiFastBroadcastExt
