import tensorflow as tf
import numpy as np


def lowres_tensor(shape, underlying_shape, offset=None, sd=None):
    """Produces a tensor paramaterized by a interpolated lower resolution tensor.
    This is like what is done in a laplacian pyramid, but a bit more general. It
    can be a powerful way to describe images.
    Args:
    shape: desired shape of resulting tensor
    underlying_shape: shape of the tensor being resized into final tensor
    offset: Describes how to offset the interpolated vector (like phase in a
      Fourier transform). If None, apply no offset. If a scalar, apply the same
      offset to each dimension; if a list use each entry for each dimension.
      If a int, offset by that much. If False, do not offset. If True, offset by
      half the ratio between shape and underlying shape (analagous to 90
      degrees).
    sd: Standard deviation of initial tensor variable.
    Returns:
    A tensor paramaterized by a lower resolution tensorflow variable.
    """
    sd = sd or 0.01
    init_val = sd*np.random.randn(*underlying_shape).astype("float32")
    underlying_t = tf.Variable(init_val)
    t = resize_bilinear_nd(underlying_t, shape)
    if offset is not None:
        # Deal with non-list offset
        if not isinstance(offset, list):
            offset = len(shape)*[offset]
        # Deal with the non-int offset entries
        for n in range(len(offset)):
            if offset[n] is True:
                offset[n] = shape[n]/underlying_shape[n]/2
            if offset[n] is False:
                offset[n] = 0
            offset[n] = int(offset[n])
        # Actually apply offset by padding and then croping off the excess.
        padding = [(pad, 0) for pad in offset]
        t = tf.pad(t, padding, "SYMMETRIC")
        begin = len(shape)*[0]
        t = tf.slice(t, begin, shape)
    return t


def laplacian_pyramid(shape, n_levels=4, sd=None):
    """Simple laplacian pyramid paramaterization of an image.
    For more flexibility, use a sum of lowres_tensor()s.
    Args:
    shape: shape of resulting image, [batch, width, height, channels].
    n_levels: number of levels of laplacian pyarmid.
    sd: standard deviation of param initialization.
    Returns:
    tensor with shape from first argument.
    """
    batch_dims = shape[:-3]
    w, h, ch = shape[-3:]
    pyramid = 0
    for n in range(n_levels):
        k = 2**n
        pyramid += lowres_tensor(shape, batch_dims + [w // k, h // k, ch], sd=sd)
    return pyramid
