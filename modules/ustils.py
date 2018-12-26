import numpy as np
from .error import *

def img2blocks(img, size):
    """ convert image to block list, row first
    """
    h, w = img.shape
    if h % size != 0 or w % size != 0:
        raise ShapeError
    block_list = []
    for i in range(w // size):
        x = i * size
        for j in range(h // size):
            y = j * size
            block = img[y: y+size, x: x+size]
            block_list.append(block)
    return block_list


def blocks2img(block_list, block_size, img_shape, dtype=None):
    """ convert blocks to image, row first
    """
    if dtype is None:
        dtype = block_list[0].dtype
    h, w = img_shape
    if w * h != len(block_list) * (block_size ** 2) \
        or h % block_size != 0 or w % block_size !=0:
        raise ShapeError
    image = np.empty(img_shape, dtype)
    idx = 0
    for i in range(h // block_size):
        x = i * block_size
        for j in range(w // block_size):
            y = j * block_size
            image[y: y+size, x: x+size] = block_list[idx]
            idx += 1
    return image


def mvlist2img(mvlist, img_shape):
    """ convert motion vector to 2 images, first represent x, second represent y
    Args:
        mv_list: motion vector list, row first
        img_shape: origion image shape divide block size
    Returns:
        (imgx, imgy): two numpy array with shape `img_shape` and type `int32`
    """
    h, w = img_shape
    if w * h != len(mvlist):
        raise ShapeError
    imgx = np.empty(img_shape, np.int32)
    imgy = np.empty(img_shape, np.int32)
    idx = 0
    for i in range(h):
        for j in range(w):
            imgx[i, j], imgy[i, j] = mvlist[idx]
            idx += 1
    return imgx, imgy

def img2mvlist(imgx, imgy):
    """ convert motion vector image to list
    """
    # TODO: not implenment
    raise NotImplementedError