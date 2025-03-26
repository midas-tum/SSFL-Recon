
import merlintf
import numpy as np


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
