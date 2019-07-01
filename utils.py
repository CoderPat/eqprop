import jax.numpy as np
from jax import vmap


def vmap_sum(fun, in_axis):
    tmp_fn = vmap(fun, in_axis, 0)
    return lambda *args: np.sum(tmp_fn(*args))

def vmap_mean(fun, in_axis=0, out_axis=0):
    tmp_fn = vmap(fun, in_axis, out_axis)

    def mean(*args):
        out = tmp_fn(*args)
        if isinstance(out, list):
            return [np.mean(o, axis=out_axis) for o in out]
        else:
            return np.mean(out, axis=out_axis)

    return mean
