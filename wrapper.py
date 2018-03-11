import os
import numpy as np

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    from dream import Dreamer

    rop_tensorflow_model = './rop_classification_model.pb'

    # gan_dreamer = Dreamer(input_model=None,
    #                         keras_model=rop_keras_model,
    #                         keras_weights=rop_keras_weights,
    #                         output_keras_conversion=rop_tensorflow_model,
    #                         output_folder='./pages_convs_andrew',
    #                         delete_output_folder=True,
    #                         ops_types=['dis_n_conv_2_8/Conv2D'],
    #                         channels=3,
    #                         image_size=1024,
    #                         dream_mode='deepdream',
    #                         specific_filters=[19],
    #                         specific_layers=['dis_n_conv_2_8'],
    #                         iterations=35,
    #                         input_base_image='me.jpg',
    #                         save_color=True)

    gan_dreamer = Dreamer(input_model=None,
                            keras_model=rop_keras_model,
                            keras_weights=rop_keras_weights,
                            output_keras_conversion=rop_tensorflow_model,
                            output_folder='./rop_classification_feature_vec',
                            delete_output_folder=True,
                            ops_types=['pool5/7x7_s2/AvgPool'],
                            ops_prefixes=[''],
                            channels_last=False,
                            input_tensor_name='input_1',
                            channels=3,
                            image_size=224,
                            dream_mode='laplace',
                            specific_filters=None,
                            specific_layers=None,
                            iterations=50,
                            input_base_image=None,
                            save_color=True)

    gan_dreamer.open_model()
    # gan_dreamer.find_layers()
    gan_dreamer.dream_image()