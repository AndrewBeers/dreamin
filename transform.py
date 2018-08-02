import tensorflow as tf

def jitter(d, seed=None, channels_last=True):

    def inner(t_image):
        t_image = tf.convert_to_tensor(t_image, preferred_dtype=tf.float32)
        t_shp = tf.shape(t_image)

        if channels_last:
            # Channels Last
            # [batch, 224, 224, 3]
            crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - d, t_shp[-1:]], 0)
            crop = tf.random_crop(t_image, crop_shape, seed=seed)
            shp = t_image.get_shape().as_list()
            mid_shp_changed = [shp[-3] - d if shp[-3] is not None else None, shp[-2] - d if shp[-3] is not None else None]
            crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])

        else:
        # Channels First
        # [batch, 3, 224, 224]
            crop_shape = tf.concat([t_shp[-4:-3], t_shp[-3:-2], t_shp[2:] - d], 0)
            crop = tf.random_crop(t_image, crop_shape, seed=seed)
            shp = t_image.get_shape().as_list()
            mid_shp_changed = [shp[-2] - d if shp[-2] is not None else None, shp[-1] - d if shp[-2] is not None else None]
            crop.set_shape(shp[-4:-3] + shp[-3:-2] + mid_shp_changed)

        return crop

    return inner

def pad(w, mode="REFLECT", constant_value=0.5, channels_last=True):

    def inner(t_image):
        if constant_value == "uniform":
            constant_value_ = tf.random_uniform([], 0, 1)
        else:
            constant_value_ = constant_value

        if channels_last:
            return tf.pad(t_image, [(0,0), (w,w), (w,w), (0,0)], mode=mode, constant_values=constant_value_)
        else:
            return tf.pad(t_image, [(0,0), (0,0), (w,w), (w,w)], mode=mode, constant_values=constant_value_)

    return inner

def random_scale(scales, seed=None):

    def inner(t):
        t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
        scale = _rand_select(scales, seed=seed)
        shp = tf.shape(t)
        scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")
        return tf.image.resize_bilinear(t, scale_shape)

    return inner

def random_rotate(angles, units="degrees", seed=None):
    def inner(t):
        t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
        angle = _rand_select(angles, seed=seed)
        angle = _angle2rads(angle, units)
        return tf.contrib.image.rotate(t, angle)
    return inner

def _rand_select(xs, seed=None):

    rand_n = tf.random_uniform((), 0, len(xs), "int32", seed=seed)
    return tf.constant(xs)[rand_n]

def _angle2rads(angle, units):
    angle = tf.cast(angle, "float32")
    if units.lower() == "degrees":
        angle = 3.14*angle/180.
    elif units.lower() in ["radians", "rads", "rad"]:
        angle = angle
    return angle