import numpy as np
import tensorflow as tf
import keras
import os

import lucid.modelzoo.vision_models as models
import lucid.modelzoo.vision_base as vision_base
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

from lucid.modelzoo.util import load_text_labels, load_graphdef, forget_xy
from custom_layers import PoolHelper, LRN

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

keras_model = '/mnt/jk489/James/plus_classification/results_with_200_excluded/tf_resnet_test/classification/Split0_Model_128features/model_arch.json'
keras_weights = '/mnt/jk489/James/plus_classification/results_with_200_excluded/tf_resnet_test/classification/Split0_Model_128features/best_weights.h5'
tf_model = 'rop_classification_model.pb'

# json_file = open(keras_model, 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = keras.models.model_from_json(loaded_model_json, custom_objects={'PoolHelper':PoolHelper, 'LRN':LRN})
# model.load_weights(keras_weights)

tf.saved_model.loader.load(sess, ['rop_masks'], '/home/local/PARTNERS/azb22/Github/dreamTexts/saved_pages/9')

# for name in graph.get_operations():
#     try:
#         print(name.name)
#     except:
#         continue

# fd = dg

init = tf.global_variables_initializer()
sess.run(init)

class PbModel(vision_base.Model):
    model_path = '/home/local/PARTNERS/azb22/Github/dreamTexts/saved_rop_masks/8/'
    labels_path = None
    image_shape = [1024,1024,4]
    image_value_range = (-117, 255-117)
    input_name = 'discriminator/discriminator_input:0'

    def __init__(self):
        self.graph_def = None
        if self.labels_path is not None:
            self.labels = load_text_labels(self.labels_path)

    def load_graphdef(self):
        self.graph_def = graph.as_graph_def()

    def post_import(self, scope):
        pass

    def create_input(self, t_input=None, forget_xy_shape=True):
        """Create input tensor."""
        if t_input is None:
            t_input = tf.placeholder(tf.float32, self.image_shape)
        t_prep_input = t_input
        if len(t_prep_input.shape) == 3:
            t_prep_input = tf.expand_dims(t_prep_input, 0)
        if forget_xy_shape:
            t_prep_input = forget_xy(t_prep_input)
        lo, hi = self.image_value_range
        t_prep_input = lo + t_prep_input * (hi-lo)
        return t_input, t_prep_input

    def import_graph(self, t_input=None, scope='import', forget_xy_shape=True):
        """Import model GraphDef into the current graph."""
        graph = tf.get_default_graph()
        assert graph.unique_name(scope, False) == scope, (
        'Scope "%s" already exists. Provide explicit scope names when '
        'importing multiple instances of the model.') % scope
        t_input, t_prep_input = self.create_input(t_input, forget_xy_shape)
        print(t_input, t_prep_input)
        tf.import_graph_def(self.graph_def, {self.input_name: t_prep_input}, name=scope)
        self.post_import(scope)

# lucid_model = vision_base.Model()
# lucid_model.model_path = tf_model
lucid_model = PbModel()
lucid_model.load_graphdef()


temp_graph_def = graph.as_graph_def()

# fd = dg

# print(lucid_model.graph_def)

# model = models.InceptionV1().load_graphdef()
# print(model)
# print(dir(model))

obj = objectives.channel("discriminator/dis_n_conv_1_4/Conv2D", 2)
param_f = lambda: tf.concat([
    param.rgb_sigmoid(param.naive([1, 128, 128, 3])),
    param.fancy_colors(param.naive([1, 128, 128, 8])/1.3),
    param.rgb_sigmoid(param.laplacian_pyramid([1, 128, 128, 3])/2.),
    param.fancy_colors(param.laplacian_pyramid([1, 128, 128, 8])/2./1.3),
], 0)
render.render_vis(lucid_model, obj, param_f)

# _ = render.render_vis(lucid_model, "discriminator/dis_n_conv_1_4/Conv2D:0")