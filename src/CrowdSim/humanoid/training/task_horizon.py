"""Training-only future root command padding, retaining the actual mocap clock."""
import math
import numpy as np

def horizon_frame_offsets(num_envs, horizons_s):
    if type(num_envs) is not int or num_envs<1:
        raise ValueError('Positive integer actor count required')
    values=tuple(horizons_s)
    if not values or any(type(v) not in (int,float) or not math.isfinite(v)
                         or not .5<=v<=4.5 or not math.isclose(v*30,round(v*30),abs_tol=1e-9)
                         for v in values):
        raise ValueError('Finite exact30Hz horizons in .5..4.5 seconds required')
    caps=np.array([round(v*30) for v in values],dtype=np.int64)
    # The first target remains the real current clip phase, not a future frame.
    offsets=np.minimum(np.arange(0,150,15,dtype=np.int64)[None],caps[np.arange(num_envs)%len(caps),None])
    offsets.setflags(write=False)
    return offsets
