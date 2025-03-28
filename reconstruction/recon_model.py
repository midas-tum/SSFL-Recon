
import tensorflow as tf
import tensorflow.keras.backend as K

from utils.mri import MulticoilForwardOp
from utils.layers import Scalar, get_dc_layer
from utils.basic_functions import complex_scale
from .feature_assisted_unet import FeatureComplexUNet2Dt
from feature_learning.models.VICReg_feature_model import UNET2dt as feature_extractor_v
from feature_learning.models.contrastive_feature_model import UNET2dt as feature_extractor_c


def get_recon_net():
    return FeatureComplexUNet2Dt(filters=12, num_levels=2, layers_per_level=1, kernel_size_2d=(1, 5, 5),
                                 kernel_size_t=(3, 1, 1), pool_size=(2, 2, 2), activation_last=None)


class SSFL_Recon(tf.keras.Model):
    def __init__(self, num_iter, mode, feature_learning, name="SSFL_Recon", pretrained_weights=None):
        super().__init__(name=name)
        self.S_end = num_iter
        self.mode = mode
        self.FE = feature_learning
        self.dc = []
        self.tau = []
        self.recon_unet = []

        for i in range(num_iter):
            self.dc.append(get_dc_layer())
            self.tau.append(Scalar(init=0.1))
            self.recon_unet.append(get_recon_net())

        # Load pre-trained feature extractor
        if self.FE == 'contrastive':
            self.feature_extractor = feature_extractor_c(num_iter=num_iter, mode='pred')
        elif self.FE == 'vicreg':
            self.feature_extractor = feature_extractor_v(num_iter=num_iter, mode='pred')
        else:
            raise ValueError(f"Invalid feature extractor type '{self.FE}'. "
                             f"Supported options are: 'contrastive', 'vicreg'.")

        if pretrained_weights is not None:
            print('---------------------- setting pre-trained FE ----------------------')
            self.feature_extractor.load_weights(pretrained_weights)
            for layer in self.feature_extractor.layers:
                layer.trainable = False
        else:
            print(" Pretrained weights not found or not provided. Using randomly initialized encoder.")

    def update_x(self, x, y, mask, smaps, feature, num_i):
        den = self.recon_unet[num_i]([x, feature])
        x = x - complex_scale(self.tau[num_i](den), 1 / self.S_end)
        x = self.dc[num_i]([x] + [y, mask, smaps])
        return x

    def ssl_recon_loss(self, x1, x2, y1, y2, mask_1, mask_2, smaps):
        # image consistency loss between x1, x2
        # image loss between x1, x2
        x1 = tf.cast(x1, tf.complex64)
        x2 = tf.cast(x2, tf.complex64)
        diff = (x1 - x2)
        img_loss = K.mean(K.sum(tf.math.real(tf.math.conj(diff) * diff), axis=(1, 2, 3)), axis=(0, -1))

        # cross k-space loss
        x1_ksp_mask2 = MulticoilForwardOp(center=True)(x1, mask_2, smaps)
        diff = (x1_ksp_mask2 - y2)
        ksp_loss_1 = K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj(diff) * diff) + 1e-9)))

        x2_ksp_mask1 = MulticoilForwardOp(center=True)(x2, mask_1, smaps)
        diff = (x2_ksp_mask1 - y1)
        ksp_loss_2 = K.mean(K.sum(tf.sqrt(tf.math.real(tf.math.conj(diff) * diff) + 1e-9)))

        ksp_loss = ksp_loss_1 + ksp_loss_2
        return img_loss, ksp_loss

    def call(self, inputs):
        if self.mode == 'train':
            x1, y1, mask_1, x2, y2, mask_2, smaps = inputs

            # get pre-learned features
            feature_representations_1 = self.feature_extractor([x1, y1, mask_1, smaps])
            feature_representations_2 = self.feature_extractor([x2, y2, mask_2, smaps])

            # feature-assisted self-supervised reconstruction
            for i in range(self.S_end):
                x1 = self.update_x(x1, y1, mask_1, smaps, feature_representations_1[i], i)
                x2 = self.update_x(x2, y2, mask_2, smaps, feature_representations_2[i], i)
            img_loss, ksp_loss = self.ssl_recon_loss(x1, x2, y1, y2, mask_1, mask_2, smaps)
            output = tf.stack((img_loss, ksp_loss), axis=0)
            return output

        if self.mode == 'pred':
            x, y, mask, smaps = inputs
            feature_representations = self.feature_extractor([x, y, mask, smaps])
            for i in range(self.S_end):
                x = self.update_x(x, y, mask, smaps, feature_representations[i], i)
            return x

