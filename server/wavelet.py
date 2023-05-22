import numpy as np
import pywt
import cv2
from PIL import Image


def w2d(img, mode='haar', level=1):
    # Convert PIL image to numpy array
    imArray = np.array(img)

    # Datatype conversions
    # Convert to grayscale
    imArray = np.array(Image.fromarray(imArray).convert('L'))
    # Convert to float
    imArray = np.float32(imArray) / 255.0

    # Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H