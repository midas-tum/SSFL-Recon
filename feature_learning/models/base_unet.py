
import merlintf
import numpy as np
import tensorflow as tf


class BaseComplexUNet2Dt(tf.keras.Model):
    def __init__(self, filters=64, num_levels=4, layers_per_level=2, activation='ModReLU', activation_last=None,
                 kernel_size_2d=(1, 3, 3), kernel_size_t=(3, 1, 1), pool_size=(2, 2, 2), use_bias=True,
                 output_channel=1, name='BaseComplexUNet2Dt', padding='none', **kwargs):
        super().__init__(name=name)

        self.filters = filters
        self.num_levels = num_levels
        self.layers_per_level = layers_per_level
        self.kernel_size_2d = kernel_size_2d
        self.kernel_size_t = kernel_size_t
        self.pool_size = pool_size
        self.use_bias = use_bias
        self.activation = activation
        self.activation_last = activation_last
        self.out_cha = output_channel
        self.padding = padding

        if 'in_shape' in kwargs:
            self.use_padding = self.is_padding_needed(kwargs.get('in_shape'))
        else:
            self.use_padding = self.is_padding_needed()  # in_shape at build time not known

        # Complex ops from merlintf
        self.conv_layer = merlintf.keras.layers.ComplexConvolution('3D')
        self.crop_layer = merlintf.keras.layers.Cropping('3D')
        self.down_layer = merlintf.keras.layers.MagnitudeMaxPooling('3D')
        self.up_layer = merlintf.keras.layers.ComplexConvolutionTranspose('3D')
        self.norm_layer = merlintf.keras.layers.complex_norm.ComplexInstanceNormalization
        self.activation_layer = merlintf.keras.layers.complex_act.ModReLU

        if self.padding.lower() == 'zero':
            self.pad_layer = merlintf.keras.layers.ZeroPadding('3D')
        else:
            self.pad_layer = merlintf.keras.layers.Pad3D

    def conv_block(self, filters, kernel_size, **kwargs):
        """Create one convolution block: convolution + normalization + activation."""
        return [
            self.conv_layer(filters, kernel_size, activation=None, padding='same', use_bias=self.use_bias, **kwargs),
            self.norm_layer(),
            self.activation_layer()
        ]

    def build_encoder(self):
        stage = []
        for ilevel in range(self.num_levels):
            level = []
            filters = self.filters * (2 ** ilevel)
            for ilayer in range(self.layers_per_level):
                level += self.conv_block(filters, self.kernel_size_2d)
                level += self.conv_block(filters, self.kernel_size_t)
            level.append(self.down_layer(pool_size=self.pool_size))
            stage.append(level)
        return stage

    def build_bottleneck(self, **kwargs):
        stage = []
        filters = self.filters * (2 ** (self.num_levels))
        for ilayer in range(self.layers_per_level):
            stage += self.conv_block(filters, self.kernel_size_2d)
            stage += self.conv_block(filters, self.kernel_size_t)
        stage.append(self.up_layer(self.filters * (2 ** (self.num_levels - 1)), (1, 1, 1), strides=self.pool_size,
                                   use_bias=self.use_bias, activation=self.activation, padding='same', **kwargs))
        return stage

    def build_decoder(self, **kwargs):
        stage = []
        for ilevel in range(self.num_levels-1, -1, -1):
            level = []
            filters = self.filters * (2 ** ilevel)
            for ilayer in range(self.layers_per_level):
                level += self.conv_block(filters, self.kernel_size_t)
                level += self.conv_block(filters, self.kernel_size_2d)
            if ilevel > 0:
                level.append(self.up_layer(self.filters * (2 ** (ilevel - 1)), (1, 1, 1), strides=self.pool_size,
                                           use_bias=self.use_bias, activation=self.activation, padding='same', **kwargs))
            stage.append(level)
        return stage

    def build_output_layer(self, **kwargs):
        return self.conv_layer(self.out_cha, (1, 1, 1), use_bias=self.use_bias, activation=self.activation_last,
                                     padding='same', **kwargs)

    def is_padding_needed(self, in_shape=None):
        # in_shape (excluding batch and channel dimension!)
        if not self.padding.lower() == 'none' and in_shape is None:
            print(
                'merlintf.keras.models.unet: Check if input padding/output cropping is needed. No input shape specified, potentially switching to eager mode execution. Please provide input_shape by calling: model.is_padding_needed(input_shape)')
        if in_shape is None:  # input shape not specified or dynamically varying
            self.use_padding = True
            self.pad = None
            self.optotf_pad = None
        else:  # input shape specified
            self.pad, self.optotf_pad = self.calculate_padding(in_shape)
            if np.all(np.asarray(self.pad) == 0):
                self.use_padding = False
            else:
                self.use_padding = True
        if self.padding.lower() == 'force_none':
            self.use_padding = False
            self.pad = None
            self.optotf_pad = None
        if self.use_padding:
            if self.padding.lower() == 'none':
                self.padding = 'zero'  # default padding
            print('Safety measure: Enabling input padding and output cropping!')
            print('!!! Compile model with model.compile(run_eagerly=True) !!!')
        return self.use_padding

    def calculate_padding(self, in_shape):
        in_shape = np.asarray(in_shape)
        n_dim = merlintf.keras.utils.get_ndim('3D')
        if len(in_shape) > n_dim:
            in_shape = in_shape[:n_dim]
        factor = np.power(self.pool_size, self.num_levels)
        paddings = np.ceil(in_shape / factor) * factor - in_shape
        pad = []
        optotf_pad = []
        for idx in range(n_dim):
            pad_top = paddings[idx].astype(np.int) // 2
            pad_bottom = paddings[idx].astype(np.int) - pad_top
            optotf_pad.extend([pad_top, pad_bottom])
            pad.append((pad_top, pad_bottom))
        return tuple(pad), optotf_pad[::-1]

    def calculate_padding_tensor(self, tensor):
        # calculate pad size
        # ATTENTION: input shape calculation with tf.keras.fit() ONLY possible in eager mode because of NoneType defined shapes! -> Force eager mode execution
        imshape = tensor.get_shape().as_list()
        if tf.keras.backend.image_data_format() == 'channels_last':  # default
            imshapenp = np.array(imshape[1:len(self.pool_size) + 1]).astype(float)
        else:  # channels_first
            imshapenp = np.array(imshape[2:len(self.pool_size) + 2]).astype(float)

        return self.calculate_padding(imshapenp)

    def apply_padding(self, inputs):
        if self.use_padding:
            if self.pad is None:  # input shape cannot be determined or fixed before compile
                pad, optotf_pad = self.calculate_padding_tensor(inputs)
            else:
                pad = self.pad  # local variable to avoid permanent storage of fixed pad value in case of dynamic input shapes
                optotf_pad = self.optotf_pad
            if self.padding.lower() == 'zero':
                x = self.pad_layer(pad)(inputs)
            else:
                x = self.pad_layer(inputs, optotf_pad, self.padding)
        else:
            x = inputs
        return x

    def apply_crop(self, inputs, x):
        if self.use_padding:
            if self.pad is None:  # input shape cannot be determined or fixed before compile
                pad, _ = self.calculate_padding_tensor(inputs)
            else:
                pad = self.pad
            x = self.crop_layer(pad)(x)
        return x
