
import tensorflow as tf
from .mlp import get_MLP
from .encoder_unet import FeatureEncoderUNet
from .decoder_unet import FeatureDecoderUNet
from utils.layers import Scalar, get_dc_layer


def get_encoder():
    return FeatureEncoderUNet(filters=12, num_levels=2, layers_per_level=1,
                              kernel_size_2d=(1, 5, 5), kernel_size_t=(3, 1, 1),
                              pool_size=(2, 2, 2), activation_last=None)

def get_decoder():
    return FeatureDecoderUNet(filters=12, num_levels=2, layers_per_level=1,
                              kernel_size_2d=(1, 5, 5), kernel_size_t=(3, 1, 1),
                              pool_size=(2, 2, 2), activation_last=None)


class UNET2dt(tf.keras.Model):
    def __init__(self, num_iter=3, mode='train', name='UNET2dt'):
        super().__init__(name=name)
        # Create lists to store network components for all iterations
        self.p = []  # MLP
        self.dc = []  # data consistency layer
        self.tau = []  # scalar, controlling the regularization strength
        self.encoder = []
        self.decoder = []

        for _ in range(num_iter):
            self.p.append(get_MLP())
            self.dc.append(get_dc_layer())
            self.tau.append(Scalar(init=0.1))
            self.encoder.append(get_encoder())
            self.decoder.append(get_decoder())

        self.mode = mode
        self.S_end = num_iter

    def call(self, inputs):
        if self.mode == 'train':
            x1, x2, x3, y1, y2, y3, mask1, mask2, mask3, smaps1, smaps2, smaps3 = inputs

            x_all = [x1, x2, x3]
            y_all = [y1, y2, y3]
            mask_all = [mask1, mask2, mask3]
            smaps_all = [smaps1, smaps2, smaps3]

            all_features = []
            for i in range(self.S_end):
                for j in range(3):  # x1, x2, x3
                    x_all[j], feature = self.feature_train(x_all[j], y_all[j], mask_all[j], smaps_all[j], i)
                    all_features.append(feature)

            output = tf.stack(all_features, axis=0)
            return output

        if self.mode == 'pred':
            x, y, mask, smap = inputs
            feature1, _ = self.feature_pred(x, y, mask, smap, num_i=0)
            feature2, _ = self.feature_pred(x, y, mask, smap, num_i=1)
            feature3, _ = self.feature_pred(x, y, mask, smap, num_i=2)
            return feature1, feature2, feature3

    def feature_train(self, x, y, mask, smaps, num_i):
        r, xforward = self.encoder[num_i](x)
        # r: feature representation after encoder
        # xforward: skip connection outputs from each encoder level (for decoder concatenation)

        # transform to real-valued inputs for MLP
        r_MLP = tf.squeeze(r, axis=0)
        r_MLP_two_channel = tf.concat((tf.math.real(r_MLP), tf.math.imag(r_MLP)), axis=-1)
        for ilayer in self.p[num_i]:
            r_MLP_two_channel = ilayer(r_MLP_two_channel)  # embedding for loss calculation

        if num_i != (self.S_end - 1):  # go through the decoder except the last iteration
            # decoder
            out_decoder = self.decoder[num_i]([x, r, xforward])

            # residual connection (residual UNet)
            x = x - out_decoder * self.tau[num_i](1.0 / self.S_end)

            # data consistency
            x = self.dc[num_i]([x, y, mask, smaps])

        return x, r_MLP_two_channel


    def feature_pred(self, x, y, mask, smaps, num_i):
        h_encoder, xforward = self.encoder[num_i](x)
        h_decoder = self.decoder[num_i]([x, h_encoder, xforward])
        x = x - h_decoder * self.tau[num_i](1.0 / self.S_end)
        x = self.dc[num_i]([x, y, mask, smaps])
        return h_encoder, x
