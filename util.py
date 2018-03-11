import numpy as np
import scipy

def add_parameter(class_object, kwargs, parameter, default=None):
    if parameter in kwargs:
        setattr(class_object, parameter, kwargs.get(parameter))
    else:
        setattr(class_object, parameter, default)

def imsave(images, size, path):
    if images.shape[-1] > 4:
        for channel in xrange(4):
            new_path = path[0:-4] + '_' + str(channel) + '.png'
            scipy.misc.imsave(new_path, merge(images[..., channel][..., np.newaxis], size))
        return scipy.misc.imsave(path, merge(images[..., 4:], size))
    elif images.shape[-1] == 3:
        return scipy.misc.imsave(path, merge(images, size))
    elif images.shape[-1] == 1:
        scipy.misc.imsave(path, np.squeeze(merge(images[:,:,:,0][:,:,:,np.newaxis], size, channels=1)))
    else:
        scipy.misc.imsave(path, merge(images[:,:,:,:3], size))
        new_path = path[0:-4] + '_mask.png'
        return scipy.misc.imsave(new_path, np.squeeze(merge(images[:,:,:,3][:,:,:,np.newaxis], size, channels=1)))

def save_image(data, image_path, color=False):
    if color:
        scipy.misc.imsave(image_path, data[0,...,0:3])
        if data.shape[-1] > 3:
            for channel in range(3, data.shape[-1]):
                new_path = image_path[0:-4] + '_' + str(channel) + '.png'
                scipy.misc.imsave(new_path, data[0,...,channel])
    elif data.shape[-1] == 4:
        scipy.misc.imsave(image_path, data[0,:,:,:3])
        new_path = image_path[0:-4] + '_mask.png'
        return scipy.misc.imsave(new_path, data[0,...,3])
    else:
        scipy.misc.imsave(image_path, data[0,...,0])
        for channel in range(1, data.shape[-1]):
            new_path = image_path[0:-4] + '_' + str(channel) + '.png'
            scipy.misc.imsave(new_path, data[0,...,channel])
