
import numpy as np
import tensorflow as tf


class DummyCINE2DDataset(tf.keras.utils.Sequence):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.kspace_shape = (1, 25, 176, 132, 15)
        self.image_shape = (1, 25, 176, 132, 1)
        self.mask_shape = (1, 25, 1, 132, 1)
        self.smaps_shape = (1, 1, 176, 132, 15)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        def rand_complex(shape):
            real = np.random.rand(*shape).astype(np.float32)
            imag = np.random.rand(*shape).astype(np.float32)
            return tf.complex(real, imag)

        masked_img_1 = rand_complex(self.image_shape)
        masked_img_2 = rand_complex(self.image_shape)
        masked_img_3 = rand_complex(self.image_shape)

        masked_kspace_1 = rand_complex(self.kspace_shape)
        masked_kspace_2 = rand_complex(self.kspace_shape)
        masked_kspace_3 = rand_complex(self.kspace_shape)

        mask_1 = np.random.randint(0, 2, size=self.mask_shape).astype(np.float64)
        mask_2 = np.random.randint(0, 2, size=self.mask_shape).astype(np.float64)
        mask_3 = np.random.randint(0, 2, size=self.mask_shape).astype(np.float64)

        smaps_1 = rand_complex(self.smaps_shape)
        smaps_2 = rand_complex(self.smaps_shape)
        smaps_3 = rand_complex(self.smaps_shape)

        label = tf.zeros_like(masked_img_1)

        return [masked_img_1, masked_img_2, masked_img_3,
                masked_kspace_1, masked_kspace_2, masked_kspace_3,
                mask_1, mask_2, mask_3,
                smaps_1, smaps_2, smaps_3], label
