import os
import numpy as np

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    from dream import Dreamer

    # gan_dreamer = Dreamer(input_model=pages_model,
    #                         keras_model=rop_keras_model,
    #                         keras_weights=rop_keras_weights,
    #                         output_keras_conversion=rop_tensorflow_model,
    #                         output_folder='./PAGES_model_vanilla_naive',
    #                         delete_output_folder=True,
    #                         ops_types=['dis_n_conv_1_4/Conv2D'],
    #                         channels=3,
    #                         image_size=1024,
    #                         dream_mode='laplace',
    #                         specific_layers=['dis_n_conv_1_4'],
    #                         iterations=99,
    #                         save_color=True,                          
    #                         pad_level=2,
    #                         multiscale=False,
    #                         transforms=['jitter', 'jitter', 'pad'],
    #                         regularization='laplace')

    gan_dreamer = Dreamer(input_model=None,
                            keras_model=mobilenet_keras_model,
                            keras_weights=mobilenet_keras_weights,
                            output_keras_conversion=rop_tensorflow_model,
                            output_folder='./rop_mobilenet',
                            delete_output_folder=True,
                            ops_types=['dense_1/Relu'],
                            ops_prefixes=[''],
                            channels_last=True,
                            input_tensor_name='input_1',
                            channels=3,
                            image_size=224,
                            dream_mode='laplace',
                            specific_filters=[3,39,46,55,68,84,87],
                            specific_layers=None,
                            iterations=199,
                            input_base_image=None,
                            save_color=True,
                            multiscale=False,
                            transforms=['random_rotate', 'jitter', 'pad'],
                            regularization_mode='laplace',
                            optimize_step=1,
                            tensorboard_output='./model_tensorboard')

    gan_dreamer.open_model()
    # gan_dreamer.get_weights(['dropout'])
    # gan_dreamer.find_layers(contains=[''])
    gan_dreamer.dream_image()