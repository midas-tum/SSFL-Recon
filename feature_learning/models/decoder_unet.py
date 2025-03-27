
import tensorflow as tf
from .base_unet import BaseComplexUNet2Dt


class FeatureDecoderUNet(BaseComplexUNet2Dt):
    def __init__(self, num_levels=2, **kwargs):
        super().__init__(num_levels=num_levels, name='FeatureDecoderUNet', **kwargs)
        self.num_levels = num_levels
        self.bottleneck = self.build_bottleneck()
        self.decoder = self.build_decoder()
        self.out_layer = self.build_output_layer()

    def call(self, inputs):
        input, feature, xforward = inputs
        x = feature

        # bottleneck
        for op in self.bottleneck:
            if op is not None:
                x = op(x)

        # decoder
        for ilevel in range(self.num_levels - 1, -1, -1):
            x = tf.keras.layers.concatenate([x, xforward[ilevel]])
            for op in self.decoder[self.num_levels - 1 - ilevel]:
                if op is not None:
                    x = op(x)

        # output convolution
        x = self.out_layer(x)
        x = self.apply_crop(inputs=input, x=x)
        return x
