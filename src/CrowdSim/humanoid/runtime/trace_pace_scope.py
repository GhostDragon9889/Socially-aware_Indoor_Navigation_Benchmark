"""Release our completed control frame before native simulator teardown.

Python3.12 frame.clear() releases locals in a completed frame and refuses an
executing frame. Only this runner's exact code object is allowed. This does not
clear SDK frames, change simulation data, or replace normal process shutdown.
https://docs.python.org/3.12/reference/datamodel.html#frame-objects
"""

import gc
import types


def clear_completed_physics_frame(owner, function):
    frame = owner.physics_frame
    if type(frame) is not types.FrameType or frame.f_code is not function.__code__:
        raise ValueError("Only this owner's exact physics function frame can be cleared")
    name = frame.f_code.co_name
    frame.clear()  # Raises RuntimeError for a running frame; never suppressed.
    # CPython3.12 can retain its materialised f_locals dictionary after clear.
    # Clear only this completed frame's own map as well (including any outside
    # references to the same map), then verify all tracked objects have died.
    frame.f_locals.clear()
    owner.physics_frame = None
    del frame
    gc.collect()
    return {"function": name, "owned_completed_frame_cleared": True}
