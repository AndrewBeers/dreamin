# boilerplate code
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import keras
import json
import shutil
import sys
import csv

from keras import backend as K
from PIL import Image
from functools import partial
from io import BytesIO
from util import add_parameter, save_image
from collections import OrderedDict
from scipy.misc import imresize
from shutil import rmtree
from transform import jitter, pad, random_scale, random_rotate

from custom_layers import PoolHelper, LRN

keras.backend.set_learning_phase(0)

class Dreamer(object):

    def __init__(self, **kwargs):

        # File Parameters
        add_parameter(self, kwargs, 'input_model', None)
        add_parameter(self, kwargs, 'input_base_image', None)
        add_parameter(self, kwargs, 'output_folder', None)
        add_parameter(self, kwargs, 'delete_output_folder', False)

        # Keras Parameters
        add_parameter(self, kwargs, 'keras_model', None)
        add_parameter(self, kwargs, 'keras_weights', None)
        add_parameter(self, kwargs, 'output_keras_conversion', None)

        # Model Parameters
        add_parameter(self, kwargs, 'channels_last', True)
        add_parameter(self, kwargs, 'channels', 5)
        add_parameter(self, kwargs, 'image_size', 256)

        # Layer Parameters
        add_parameter(self, kwargs, 'input_tensor_name', 'discriminator/discriminator_input')
        add_parameter(self, kwargs, 'ops_prefixes', ['discriminator/'])
        add_parameter(self, kwargs, 'ops_types', ['Conv2D'])
        add_parameter(self, kwargs, 'specific_filters', None)
        add_parameter(self, kwargs, 'specific_layers', None)

        # Dream Parameters
        add_parameter(self, kwargs, 'transforms', ['jitter', 'pad'])
        add_parameter(self, kwargs, 'pad_level', 2)
        add_parameter(self, kwargs, 'dream_mode', 'laplace')
        add_parameter(self, kwargs, 'multiscale', True)
        add_parameter(self, kwargs, 'octave_n', 3)
        add_parameter(self, kwargs, 'octave_scale', 1.4)
        add_parameter(self, kwargs, 'optimize_step', 1)
        add_parameter(self, kwargs, 'regularization_mode', 'laplace')
        add_parameter(self, kwargs, 'iterations', 200)

        # Laplace Parameters
        add_parameter(self, kwargs, 'lap_n', 4)

        # Save Parameters
        add_parameter(self, kwargs, 'save_color', False)

        self.build()

        return

    def build(self):

        if not self.multiscale:
            self.octave_n = 1

        self.laplace_kernel = np.float32([1,4,6,4,1])
        self.laplace_kernel = np.outer(self.laplace_kernel, self.laplace_kernel)
        self.laplace_kernel5x5 = self.laplace_kernel[:,:,None,None]/self.laplace_kernel.sum()*np.eye(self.channels, dtype=np.float32)

        # Should this be initialized as None? Coding etiquette..
        self.layer_dict = OrderedDict()

        if self.input_base_image is not None:
            self.input_base_image = np.asarray(Image.open(self.input_base_image), dtype='uint8')
            # self.input_base_image = imresize(self.input_base_image, (self.image_size, self.image_size))

            if self.input_base_image.shape[-1] < self.channels:
                for i in range(self.channels - self.input_base_image.shape[-1]):
                    self.input_base_image = np.concatenate((self.input_base_image, self.input_base_image[...,0][...,np.newaxis]), axis=2)

            self.input_base_image = self.input_base_image[np.newaxis,...] / 127.5 - 1

        return

    def open_model(self):

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)

        if self.input_model is not None:
            tf.saved_model.loader.load(self.sess, ['rop_masks'], self.input_model)
        elif self.keras_model is not None:
            self.load_keras_model()

        for op in self.graph.get_operations():
            if op.name.endswith(tuple(self.ops_types)) and op.name.startswith(tuple(self.ops_prefixes)):
                if self.graph.get_tensor_by_name(op.name + ':0').get_shape() != ():
                    if self.channels_last:
                        self.layer_dict[op.name] = {'tensor': op, 'feature_num': int(self.graph.get_tensor_by_name(op.name + ':0').get_shape()[-1])}
                    else:
                        self.layer_dict[op.name] = {'tensor': op, 'feature_num': int(self.graph.get_tensor_by_name(op.name + ':0').get_shape()[-1])}

        if True:
            for name in self.layer_dict:
                print(name)

        self.lap_norm_func = self.tffunc(np.float32)(partial(self.lap_normalize, scale_n=self.lap_n))

        self.transform_dict = {'jitter': jitter(self.pad_level),
                                'pad': pad(self.pad_level/2),
                                'random_scale': random_scale([1 + (i-5)/50. for i in range(11)]),
                                'random_rotate': random_rotate([0,90,180,270])}

        self.transforms = [self.transform_dict[t] for t in self.transforms]

        if self.transforms != []:
            self.composed_transform = self.compose(self.transforms)
        else:
            self.composed_transform = None

    def load_keras_model(self):

        json_file = open(self.keras_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = keras.models.model_from_json(loaded_model_json, custom_objects={'PoolHelper': PoolHelper, 'LRN': LRN, 'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2d': keras.applications.mobilenet.DepthwiseConv2D})

        # print(dir(model))
        # for layer in model.layers:
        #     print(layer)
        # g

        if self.keras_weights is not None:
            model.load_weights(self.keras_weights)

        return

    def find_layers(self, contains=['discriminator/']):

        for layer in self.graph.get_operations():
            if any(op_type in layer.name for op_type in contains):
                try:
                    if self.graph.get_tensor_by_name(layer.name+':0').get_shape() != ():
                        print(layer.name, self.graph.get_tensor_by_name(layer.name+':0').get_shape())
                except:
                    continue

    def get_weights(self, contains=['']):

        for layer in self.graph.get_operations():
            if any(op_type in layer.name for op_type in contains):
                try:
                    if self.graph.get_tensor_by_name(layer.name+':0').get_shape() != ():
                        print(layer.name)
                        print(self.graph.get_tensor_by_name(layer.name+':0'))
                except:
                    continue      

    def save_weights(self, weights_name):

        with open('weights.csv', 'wb') as writefile:
            csvfile = csv.writer(writefile, delimiter=',')
            for layer in self.graph.get_operations():
                if layer.name == weights_name:
                    weights_array = self.graph.get_tensor_by_name(layer.name+':0').eval()
                    for row_idx, row in enumerate(weights_array):
                        csvfile.writerow([row_idx] + row.tolist())

    def dream_image(self):

        if self.delete_output_folder and os.path.exists(self.output_folder):
            rmtree(self.output_folder)

        for folder in [self.output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.input_tensor = self.graph.get_tensor_by_name(self.input_tensor_name + ':0')
        print(self.input_tensor)

        print('Number of layers', len(self.layer_dict))
        # print('Total number of feature channels:', sum(feature_nums))
        # self.find_layers()

        self.resize_func = self.tffunc(np.float32, np.int32)(self.resize)

        for layer_name in reversed(self.layer_dict.keys()):
            activated_tensor = self.grab_tensor(layer_name)
            print(layer_name, self.layer_dict[layer_name]['feature_num'])

            # Doesn't make too much sense on a layerwise basis.
            if self.specific_filters is None:
                filter_iter = range(self.layer_dict[layer_name]['feature_num'])
            else:
                filter_iter = self.specific_filters

            for filter_num in filter_iter:

                output_filename = os.path.join(self.output_folder, '_'.join([self.dream_mode, layer_name.replace('/', '_'), 'filter', str(filter_num)]) + '.png')

                if os.path.exists(output_filename):
                    continue

                if self.input_base_image is None:
                    img_noise = np.random.uniform(size=(1, self.image_size, self.image_size, self.channels), low=-1, high=1).astype(np.float32)
                else:
                    img_noise = self.input_base_image

                print('Input Tensor Shape', activated_tensor.shape)
                # activated_filter = activated_tensor[..., filter_num]
                filter_list = [119, 106, 91]
                tensor_list = []
                for i in filter_list + [filter_num]:
                    tensor_list += [activated_tensor[..., i]]
                activated_filter = tf.stack(tensor_list, axis=-1)

                print('Input Filter Shape', activated_filter.shape)

                self.render(activated_filter, img_noise, output_filename=output_filename, iterations=self.iterations)

    def render(self, optimization_tensor, input_img, output_filename='test.png', iterations=20):
        t_score = tf.reduce_mean(optimization_tensor)  # defining the optimization objective
        t_grad = tf.gradients(t_score, self.input_tensor)[0]  # behold the power of automatic differentiation!

        img = input_img.copy()

        img = self.image_optimize(img, iterations, t_grad)

        if img is not None:
            img = self.norm_visualize(img)
            img = np.clip(img, -1, 1)
            save_image(img, output_filename, color=self.save_color)

    def image_optimize(self, img, iterations, t_grad):

        renderable = True

        for octave in range(self.octave_n):
            if octave > 0:
                hw = np.float32(img.shape[1:3]) * self.octave_scale
                img = self.resize_func(img, np.int32(hw))
            for i in range(iterations):
                # img = self.lap_norm_func(img)
                # opti_img = img
                if i % 1 == 5 and self.composed_transform is not None:
                    img = self.composed_transform(img).eval()
                g = self.calc_gradient(img, t_grad, tile_size=self.image_size*2)
                # normalizing the gradient, so the same step size should work
                if np.sum(g) == 0:
                    renderable = False
                    break
                # print(np.sum(g), 'before_reg')
                g = self.regularize(g)
                # print(np.sum(g), 'after_reg')
                img += g*self.optimize_step
                print('.', end=' ')
                sys.stdout.flush()
        print('\n')
        if renderable:
            return img
        else:
            return None

    def calc_gradient(self, img, t_grad, tile_size=1024):

        if self.multiscale:
            sz = tile_size
            h, w = img.shape[1:3]
            sx, sy = np.random.randint(sz, size=2)
            img_shift = np.roll(np.roll(img, sx, 2), sy, 1)
            grad = np.zeros_like(img)
            for y in range(0, max(h-sz//2, sz),sz):
                for x in range(0, max(w-sz//2, sz),sz):
                    sub = img_shift[:,y:y+sz/2,x:x+sz/2,:]
                    if self.channels_last:
                        g = self.sess.run(t_grad, {self.input_tensor:sub})
                    else:
                        sub = np.swapaxes(sub, 1, -1)
                        g = self.sess.run(t_grad, {self.input_tensor:sub})
                        g = np.swapaxes(g, 1, -1)
                    grad[:,y:y+sz/2,x:x+sz/2,:] = g
            return np.roll(np.roll(grad, -sx, 2), -sy, 1)

        else:
            if self.channels_last:
                g = self.sess.run(t_grad, {self.input_tensor:img})
            else:
                img = np.swapaxes(img, 1, -1)
                g = self.sess.run(t_grad, {self.input_tensor:img})
                g = np.swapaxes(g, 1, -1)
            return g

    def regularize(self, g):

        if self.regularization_mode == 'naive':
            g /= g.std()+1e-8
            return g

        if self.regularization_mode == 'laplace':
            return self.lap_norm_func(g)

    def grab_tensor(self, layer):
        '''Helper for getting layer output tensor'''
        return self.graph.get_tensor_by_name(layer +':0')

    def transform_input_image(self, image):

        return

    def norm_visualize(self, a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

    def tffunc(self, *argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kwargs):
                return out.eval(dict(zip(placeholders, args)), session=kwargs.get('session'))
            return wrapper
        return wrap

    # Helper function that uses TF to resize an image
    def resize(self, img, size):
        return tf.image.resize_bilinear(img, size)

    def lap_split(self, img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, self.laplace_kernel5x5, [1, 2, 2, 1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, self.laplace_kernel5x5 * 4, tf.shape(img), [1, 2, 2, 1])
            hi = img-lo2
        return lo, hi

    def lap_split_n(self, img, n):
        '''Build Laplacian pyramid with n splits'''
        levels = []
        for i in range(n):
            img, hi = self.lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(self, levels):
        '''Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv2d_transpose(img, self.laplace_kernel5x5*4, tf.shape(hi), [1,2,2,1]) + hi
        return img

    def normalize_std(self, img, eps=1e-10):
        '''Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img/tf.maximum(std, eps)

    def lap_normalize(self, img, scale_n=4):
        '''Perform the Laplacian pyramid normalization.'''
        tlevels = self.lap_split_n(img, scale_n)
        tlevels = list(map(self.normalize_std, tlevels))
        out = self.lap_merge(tlevels)
        return out

    def compose(self, transforms):
        def inner(x):
            for transform in transforms:
                x = transform(x)
            return x
        return inner

    def commented_code(self):
        # creating TensorFlow session and loading the model
        # graph = tf.Graph()
        # sess = tf.InteractiveSession(graph=graph)
        # with tf.gfile.FastGFile(self.input_model, 'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        # t_input = tf.placeholder(np.float32, name='input') # define the input tensor
        # imagenet_mean = 0
        # t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
        # tf.import_graph_def(graph_def, {'input':t_preprocessed})

        # self.input_tensor = tf.placeholder(tf.float32, [1, 512, 512, 4], name='input')

        # with tf.gfile.FastGFile(os.path.join(self.input_model, 'saved_model.pb'), 'rb') as f:
            # graph_def = tf.GraphDef()
            # graph_def.ParseFromString(f.read())

        # octaves = []
        # for i in range(octave_n-1):
        #     hw = img.shape[1:3]
        #     print(hw)
        #     new_hw = [np.float32(x)/octave_scale for x in hw]
        #     lo = self.resize(img, np.int32(new_hw)).eval()
        #     hi = img-self.resize(lo, hw).eval()
        #     img = lo
        #     octaves.append(hi)
        
        # # generate details octave by octave
        # for octave in range(octave_n):
        #     if octave > 0:
        #         hi = octaves[-octave]
        #         img = self.resize(img, hi.shape[1:3]).eval()+hi
        #     for i in range(iterations):
        #         g = self.calc_grad_tiled(img, t_grad, tile_size=self.image_size*2)
        #         g = lap_norm_func(g)
        #         img += g*(step / (np.abs(g).mean()+1e-7))
        #         print('.',end = ' ')


        return

# render_multiscale(T(layer)[:,:,:,channel])

# render_naive(T(layer)[:,:,:,channel])

# Helper functions for TF Graph visualization

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def select(lst, *indices):
    return (lst[i] for i in indices)


# Visualizing the network graph. Be sure expand the "mixed" nodes to see their 
# internal structure. We are going to visualize "Conv2D" nodes.
# tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
# show_graph(tmp_def)
