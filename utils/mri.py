
import merlintf
import numpy as np
import tensorflow as tf
from .basic_functions import complex_scale
from tensorflow.signal import fft2d, ifft2d


def FFT2c(image):
    """
    Apply centered 2D Fourier Transform (FFT) to complex-valued MRI image data.

    This function transforms multi-coil spatial domain image data into k-space
    using centered FFT (i.e., fftshift + ifftshift).

    Args:
        image (np.ndarray): Input complex image of shape
            (nSlice, nTime, nFE, nPE, nCoil), where:
                - nSlice: number of slices
                - nTime: number of time frames
                - nFE: frequency encoding dimension (x)
                - nPE: phase encoding dimension (y)
                - nCoil: number of coils

    Returns:
        np.ndarray: Complex-valued k-space data of the same shape.
    """
    # Reorder to (slice, time, coil, x, y) for FFT
    image = np.transpose(image, axes=(0, 1, 4, 2, 3))
    axes = [3, 4]  # Apply FFT over spatial dimensions
    scale = np.sqrt(np.prod(image.shape[-2:]).astype(np.float64))

    kspace = merlintf.complex_scale(
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes),
        1 / scale
    )

    # Return to original shape: (slice, time, x, y, coil)
    return np.transpose(kspace, axes=(0, 1, 3, 4, 2))


def IFFT2c(kspace):
    """
    Apply centered 2D Inverse Fourier Transform (IFFT) to complex k-space data.

    This function reconstructs image domain data from multi-coil k-space using
    centered inverse FFT (i.e., fftshift + ifftshift).

    Args:
        kspace (np.ndarray): Input complex k-space of shape
            (nSlice, nTime, nFE, nPE, nCoil)

    Returns:
        np.ndarray: Reconstructed image of the same shape.
    """
    # Reorder to (slice, time, coil, x, y) for IFFT
    kspace = np.transpose(kspace, axes=(0, 1, 4, 2, 3))
    axes = [3, 4]
    scale = np.sqrt(np.prod(kspace.shape[-2:]).astype(np.float64))

    image = merlintf.complex_scale(
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes), axes=axes),
        scale
    )

    # Return to original shape: (slice, time, x, y, coil)
    return np.transpose(image, axes=(0, 1, 3, 4, 2))


class IFFT2c_layer(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        if len(kspace.shape) == 4:
            kspace = tf.expand_dims(kspace, axis=0)
        kspace = tf.transpose(kspace, perm=[0, 1, 4, 2, 3])
        axes = [tf.rank(kspace) - 2, tf.rank(kspace) - 1]  # 3,4
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        image = complex_scale(tf.signal.fftshift(ifft2d(tf.signal.ifftshift(kspace, axes=axes)), axes=axes), scale)
        return tf.transpose(image, perm=[0, 1, 3, 4, 2])


class FFT2c_layer(tf.keras.layers.Layer):
    def call(self, image, *args):
        if len(image.shape) == 4:
            image = tf.expand_dims(image, axis=0)
        if len(image.shape) == 6:
            image = tf.squeeze(image, axis=0)
        image = tf.transpose(image, perm=[0, 1, 4, 2, 3])
        dtype = tf.math.real(image).dtype
        axes = [tf.rank(image) - 2, tf.rank(image) - 1]  # axes have to be positive...
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        kspace = complex_scale(tf.signal.fftshift(fft2d(tf.signal.ifftshift(image, axes=axes)), axes=axes), 1/scale)
        return tf.transpose(kspace, perm=[0, 1, 3, 4, 2])


class IFFT2_layer(tf.keras.layers.Layer):
    def call(self, kspace, *args):
        kspace = tf.transpose(kspace, perm=[0, 1, 4, 2, 3])
        dtype = tf.math.real(kspace).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(kspace)[-2:]), dtype))
        image = complex_scale(ifft2d(kspace), scale)
        return tf.transpose(image, perm=[0, 1, 3, 4, 2])


class FFT2_layer(tf.keras.layers.Layer):
    def call(self, image, *args):
        image = tf.transpose(image, perm=[0, 1, 4, 2, 3])
        dtype = tf.math.real(image).dtype
        scale = tf.math.sqrt(tf.cast(tf.math.reduce_prod(tf.shape(image)[-2:]), dtype))
        kspace = complex_scale(fft2d(image), 1/scale)
        return tf.transpose(kspace, perm=[0, 1, 3, 4, 2])


class MaskKspace(tf.keras.layers.Layer):
    def call(self, kspace, mask):
        return complex_scale(kspace, mask)


class Smaps(tf.keras.layers.Layer):
    def call(self, img, smaps):
        img = tf.cast(img, dtype=tf.complex64)
        smaps = tf.cast(smaps, dtype=tf.complex64)
        return img * smaps


class SmapsAdj(tf.keras.layers.Layer):
    def call(self, coilimg, smaps):
        coilimg = tf.cast(coilimg, dtype=tf.complex64)
        smaps = tf.cast(smaps, dtype=tf.complex64)
        return tf.reduce_sum(coilimg * tf.math.conj(smaps), -1)


class MulticoilForwardOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        if center:
            self.fft2 = FFT2c_layer()
        else:
            self.fft2 = FFT2_layer()
        self.mask = MaskKspace()
        self.smaps = Smaps()

    def call(self, image, mask, smaps):
        coilimg = self.smaps(image, smaps)
        kspace = self.fft2(coilimg)
        masked_kspace = self.mask(kspace, mask)
        return masked_kspace


class MulticoilAdjointOp(tf.keras.layers.Layer):
    def __init__(self, center=False):
        super().__init__()
        self.mask = MaskKspace()
        if center:
            self.ifft2 = IFFT2c_layer()
        else:
            self.ifft2 = IFFT2_layer()
        self.adj_smaps = SmapsAdj()

    def call(self, kspace, mask, smaps):
        masked_kspace = self.mask(kspace, mask)
        coilimg = self.ifft2(masked_kspace)
        img = self.adj_smaps(coilimg, smaps)
        return tf.expand_dims(img, -1)

