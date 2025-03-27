
from .base_unet import BaseComplexUNet2Dt


class FeatureEncoderUNet(BaseComplexUNet2Dt):
    def __init__(self, num_levels=2, **kwargs):
        super().__init__(num_levels=num_levels, name='FeatureEncoderUNet', **kwargs)
        self.num_level = num_levels
        self.encoder = self.build_encoder()

    def call(self, inputs):
        x = self.apply_padding(inputs)
        xforward = []

        for ilevel in range(self.num_level):
            for iop, op in enumerate(self.encoder[ilevel]):
                if iop == len(self.encoder[ilevel]) - 1:
                    xforward.append(x)
                if op is not None:
                    x = op(x)
        return x, xforward


