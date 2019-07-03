import cv2
import numpy as np
from skimage import exposure
from face_align import AbstractAligner

def get_brightness(image): 
    '''
    Get brightness of an image
    :param image: image to get brightness
    :return brightness: brightness of an image
    '''
    mean = np.mean(image, axis=(0, 1))
    return mean[0] * 0.2126 + mean[1] * 0.7152 + mean[2] * 0.0722


def single_scale_retinex(img, sigma):
    '''
    Apply single scale retinex
    :param img: image to preprocess
    :param sigma: coefficient
    :return retinex: preprocessed image
    '''
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img) - np.log10(blur)
    return retinex


def multiscale_retinex(img, sigma_list):
    '''
    Apply single scale retinex on a list of sigma
    :param img: image to preprocess
    :param sigma_list: list of coefficient to perform single scale retinex
    :return retinex: preprocessed image
    '''
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex


def color_restoration(img, alpha, beta):
    '''
    Restore lost color
    :param img: img to restore
    :param alpha: coefficient
    :param beta: coefficient
    :return color_restoration:  restored image
    '''
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
    return color_restoration


def simplest_color_balance(img, low_clip, high_clip):
    '''
    correct underexposed images
    '''
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def _msrcr(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    '''
    Perform multiscale retinex with color restoration
    '''
    img = np.float64(img) + 1.0

    img_retinex = multiscale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplest_color_balance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def _automated_msrcr(img, sigma_list):
    '''
    Perform automated multiscale retinex with color restoration
    '''
    img = np.float64(img) + 1.0

    img_retinex = multiscale_retinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        zero_count = 0
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = ((img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) /
                                (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) *
                                255)

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def align_and_crop(frame, landmarks, aligner, desired_size, prewhitening = True):
    if not isinstance(aligner, AbstractAligner):
        raise TypeError("aligner not type")
    face_aligned = aligner.align(frame, landmarks, desired_size)
    if prewhitening:
        face_aligned = whitening(face_aligned)
    return face_aligned

def _msrcp(img, sigma_list, low_clip, high_clip):
    '''
    Perform multiscale retinex with color restoration
    '''

    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multiscale_retinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplest_color_balance(retinex, low_clip, high_clip)

    intensity1 = ((intensity1 - np.min(intensity1)) /
                  (np.max(intensity1) - np.min(intensity1)) *
                  255.0 +
                  1.0)

    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp


def default_preprocess(crop, prewhitening=True):
    '''
    The default preprocessing is whitening
    '''
    if prewhitening:
        return whitening(crop)
    return crop


def whitening(image):
    '''
    Whitening: just mean normalization
    '''
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
    y = np.multiply(np.subtract(image, mean), 1/std_adj)
    return y


def contrast_stretching(crop, prewhitening=True):
    '''
    attempt to improve the contrast in an image by `stretching'
    the range of intensity values it contains to span a desired range of values
    '''
    p2 = np.percentile(crop, 2)
    p98 = np.percentile(crop, 98)
    img = exposure.rescale_intensity(crop, in_range=(p2, p98))

    if prewhitening:
        return whitening(img)
    else:
        return img


# https://en.wikipedia.org/wiki/Histogram_equalization
def histogram_equalization(crop, prewhitening=True):
    '''
    adjust image intensities to enhance contrast
    https://prateekvjoshi.com/2013/11/22/histogram-equalization-of-rgb-images/
    '''
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    equalization = cv2.merge(channels)
    img = cv2.cvtColor(equalization, cv2.COLOR_YCR_CB2BGR)

    # whitenning
    if prewhitening:
        return whitening(img)
    else:
        return img


def gamma_correction(crop, prewhitening=True):
    '''
    Change illumination of an image
    '''
    img = exposure.adjust_gamma(crop, gamma=0.5)

    # whitening
    if prewhitening:
        return whitening(img)
    else:
        return img


def msrcr(crop, prewhitening=True):
    '''
    Perform msrcr on a list of predifine hyper-parameters
    '''
    sigma_list = [15, 80, 250]
    G = 5
    b = 25
    alpha = 125
    beta = 46
    low_clip = 0.01
    high_clip = 0.99

    img = _msrcr(crop, sigma_list, G, b, alpha, beta, low_clip, high_clip)

    # whitening
    if prewhitening:
        return whitening(img)
    return img


def auto_msrcr(crop, prewhitening=True):
    '''
    Perform auto msrcr on a list of predifine hyper-parameters
    '''
    sigma_list = [15, 80, 250]
    img = _automated_msrcr(crop, sigma_list)

    # whitening
    if prewhitening:
        return whitening(img)
    return img


def msrcp(crop, prewhitening=True):
    '''
    Perform msrcp on a list of predifine hyper-parameters
    '''
    sigma_list = [15, 80, 250]
    low_clip = 0.01
    high_clip = 0.99
    img = _msrcp(crop, sigma_list, low_clip, high_clip)

    # whitening
    if prewhitening:
        return whitening(img)
    return img


class Preprocessor(object):
    '''
    Apply preprocess algs to input image for better feature extraction

    >>> processor = Preprocessor(); \
        processor._Preprocessor__algs == default_preprocess
    True
    >>> import numpy as np; \
        a = processor.process(np.ones((13, 13, 3))); \
        a.shape
    (13, 13, 3)
    '''
    def __init__(self, algs=default_preprocess):
        self.__algs = algs

    def process(self, *args):
        '''
        Process image
        :param image: an image ready for preprocess
        :return: preprocessed image
        '''
        return self.__algs(*args)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
