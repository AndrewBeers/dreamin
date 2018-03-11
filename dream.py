# boilerplate code
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import keras
import json
import shutil

from keras import backend as K
from PIL import Image
from functools import partial
from io import BytesIO
from util import add_parameter, save_image
from collections import OrderedDict
from scipy.misc import imresize
from shutil import rmtree

from custom_layers import PoolHelper, LRN

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

        # Dream Parameters
        add_parameter(self, kwargs, 'specific_filters', None)
        add_parameter(self, kwargs, 'specific_layers', None)
        add_parameter(self, kwargs, 'dream_mode', 'laplace')
        add_parameter(self, kwargs, 'iterations', 200)

        # Save Parameters
        add_parameter(self, kwargs, 'save_color', False)

        self.build()

        return

    def build(self):

        self.laplace_kernel = np.float32([1,4,6,4,1])
        self.laplace_kernel = np.outer(self.laplace_kernel, self.laplace_kernel)
        self.laplace_kernel5x5 = self.laplace_kernel[:,:,None,None]/self.laplace_kernel.sum()*np.eye(self.channels, dtype=np.float32)

        # Should this be initialized as None? Coding etiquette..
        self.layer_dict = OrderedDict()

        if self.delete_output_folder and os.path.exists(self.output_folder):
            rmtree(self.output_folder)

        for folder in [self.output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

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
                if self.channels_last:
                    self.layer_dict[op.name] = {'tensor': op, 'feature_num': int(self.graph.get_tensor_by_name(op.name+':0').get_shape()[-1])}
                else:
                    self.layer_dict[op.name] = {'tensor': op, 'feature_num': int(self.graph.get_tensor_by_name(op.name+':0').get_shape()[-1])}
        if True:
            for name in self.layer_dict:
                print(name)

    def load_keras_model(self):

        json_file = open(self.keras_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = keras.models.model_from_json(loaded_model_json, custom_objects={'PoolHelper':PoolHelper, 'LRN':LRN})
        model.load_weights(self.keras_weights)

        # for name in self.graph.get_operations():
        #     try:
        #         print(name.name), print(self.graph.get_tensor_by_name(name.name+':0'))
        #     except:
        #         continue

        # fd = dg

        return

    def find_layers(self, contains=['discriminator/']):

        for layer in self.graph.get_operations():
            if any(op_type in layer.name for op_type in contains):
                print(layer.name)

    def dream_image(self):

        self.input_tensor = self.graph.get_tensor_by_name(self.input_tensor_name + ':0')

        print('Number of layers', len(self.layer_dict))
        # print('Total number of feature channels:', sum(feature_nums))
        self.find_layers()

        self.resize_func = self.tffunc(np.float32, np.int32)(self.resize)

        for layer_name in reversed(self.layer_dict.keys()):
            activated_tensor = self.T(layer_name)
            print(layer_name, self.layer_dict[layer_name]['feature_num'])
            for channel in range(self.layer_dict[layer_name]['feature_num']):
            # for channel in [19]:

                output_filename = os.path.join(self.output_folder, '_'.join([self.dream_mode, layer_name.replace('/', '_'), 'filter', str(channel)]) + '.png')

                if os.path.exists(output_filename):
                    continue

                if self.input_base_image is None:
                    img_noise = np.random.uniform(size=(1, self.image_size, self.image_size, self.channels), low=-1, high=1)
                else:
                    img_noise = self.input_base_image

                print(activated_tensor.shape)
                activated_filter = activated_tensor[..., channel]
                print(activated_filter.shape)

                if self.dream_mode == 'naive':
                    self.render_naive(activated_filter, img_noise, output_filename=output_filename)
                elif self.dream_mode == 'multiscale':
                    self.render_multiscale(activated_filter, img_noise, output_filename=output_filename)
                elif self.dream_mode == 'laplace':
                    self.render_lapnorm(activated_filter, img_noise, output_filename=output_filename, iter_n=self.iterations)
                elif self.dream_mode == 'deepdream':
                    self.render_deepdream(activated_filter, img_noise, output_filename=output_filename, iter_n=self.iterations)

    def T(self, layer):
        '''Helper for getting layer output tensor'''
        return self.graph.get_tensor_by_name(layer +':0')

    def render_naive(self, t_obj, img0, output_filename='test.png', iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, self.input_tensor)[0] # behold the power of automatic differentiation!
        
        img = img0.copy()
        print(img.shape)
        print(self.input_tensor)
        for i in range(iter_n):
            print('iter', i)
            g, score = self.sess.run([t_grad, t_score], {self.input_tensor:img})
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
            print(score, end = ' ')
        # img = self.visstd(img)
        img = self.visstd(img)
        print(np.unique(img))
        img = np.clip(img, -1, 1)
        save_image(img, output_filename)

    def visstd(self, a, s=0.1):
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

    def calc_grad_tiled(self, img, t_grad, tile_size=1024):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over 
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[1:3]
        print(h, w)
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 2), sy, 1)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                print(h, w, x, y, sz)
                sub = img_shift[:,y:y+sz/2,x:x+sz/2,:]
                if self.channels_last:
                    g = self.sess.run(t_grad, {self.input_tensor:sub})
                else:
                    sub = np.swapaxes(sub, 1, -1)
                    g = self.sess.run(t_grad, {self.input_tensor:sub})
                    g = np.swapaxes(g, 1, -1)
                grad[:,y:y+sz/2,x:x+sz/2,:] = g
        return np.roll(np.roll(grad, -sx, 2), -sy, 1)

    def render_multiscale(self, t_obj, img0, output_filename='test.png', iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, self.input_tensor)[0] # behold the power of automatic differentiation!
        
        img = img0.copy()
        for octave in range(octave_n):
            if octave>0:
                hw = np.float32(img.shape[1:3])*octave_scale
                print(hw)
                img = self.resize_func(img, np.int32(hw))
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad, tile_size=self.image_size*2)
                # normalizing the gradient, so the same step size should work 
                g /= g.std()+1e-8         # for different layers and networks
                img += g*step
                print('.', end = ' ')
            print('octave', octave)
        img = self.visstd(img)
        print(np.unique(img))
        img = np.clip(img, -1, 1)
        save_image(img, output_filename)

    def lap_split(self, img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):
            lo = tf.nn.conv2d(img, self.laplace_kernel5x5, [1,2,2,1], 'SAME')
            lo2 = tf.nn.conv2d_transpose(lo, self.laplace_kernel5x5*4, tf.shape(img), [1,2,2,1])
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

    def render_lapnorm(self, t_obj, img0, output_filename='test.png', iter_n=40, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, self.input_tensor)[0] # behold the power of automatic differentiation!
        # build the laplacian normalization graph
        lap_norm_func = self.tffunc(np.float32)(partial(self.lap_normalize, scale_n=lap_n))

        img = img0.copy()
        for octave in range(octave_n):
            if octave>0:
                hw = np.float32(img.shape[1:3])*octave_scale
                img = self.resize(img, np.int32(hw)).eval()
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad, tile_size=self.image_size*2)
                g = lap_norm_func(g)
                img += g*step
                print('.', end = ' ')
        img = self.visstd(img)
        print(np.unique(img))
        img = np.clip(img, -1, 1)
        save_image(img, output_filename, color=self.save_color)


    def render_deepdream(self, t_obj, img0, output_filename='test.png', iter_n=35, step=.015, octave_n=1, octave_scale=1.2, lap_n=4):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, self.input_tensor)[0] # behold the power of automatic differentiation!

        lap_norm_func = self.tffunc(np.float32)(partial(self.lap_normalize, scale_n=lap_n))

        # split the image into a number of octaves
        img = img0.copy()
        octaves = []
        for i in range(octave_n-1):
            hw = img.shape[1:3]
            print(hw)
            new_hw = [np.float32(x)/octave_scale for x in hw]
            lo = self.resize(img, np.int32(new_hw)).eval()
            hi = img-self.resize(lo, hw).eval()
            img = lo
            octaves.append(hi)
        
        # generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = self.resize(img, hi.shape[1:3]).eval()+hi
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad, tile_size=self.image_size*2)
                g = lap_norm_func(g)
                img += g*(step / (np.abs(g).mean()+1e-7))
                print('.',end = ' ')
        # img = self.visstd(img)
        print(np.unique(img))
        img = np.clip(img, -1, 1)
        save_image(img, output_filename, color=self.save_color)


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

# Visualizing the network graph. Be sure expand the "mixed" nodes to see their 
# internal structure. We are going to visualize "Conv2D" nodes.
# tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
# show_graph(tmp_def)
