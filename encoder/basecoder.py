from math import ceil
from queue import Queue

import numpy as np
from bitstream import BitStream

from ..modules.DCT import DCT_transform, reverse_DCT_transform
from ..modules.ustils import img2blocks, blocks2img, mvlist2img
from ..modules.quantizer import quantizer, dequantizer
from ..modules.entropy import to_entropy_code, from_entropy_code
from ..modules.error import *

class BaseCoder(object):
    """
    bitstream format:
        [head]
        16bits: time stamp
        32bits: h, w(just for the 1st frame)
        512bits: quantization matrix, 8x8 values with 8 bits(just for the 1st frame) NOTE: if quantization matrix is not adaptive, this item is all zero
        32bits: motion vector length, x 16bits, y 16bits
        32bits: residue length
        [body]
        motion vector: x. y
        residue
    """
    def __init__(self, img_size, blcok_size, motion_search_range,\
                 block_codebook, mv_codebook, quantize_mat,\
                 file_name, bg_buffer_size):
        """
        img_size: image shape, like (W, H)
        block_size: encode block size, should be an integer like 8
        motion_search_range: search range used in motion estimation, should be an integer like 16
        block_codebook: codebook for block entropy coding, a dictionary
        mv_codebook: codebook for mv entropy coding, a tuple of two dictionary
        quantize_mat: quantization matrix
        file_name: file name
        bg_buffer size: background buffer size
        """
        self._W, self._H = img_size
        self._block_size = blcok_size
        self._motion_search_range = motion_search_range
        self._block_codebook = block_codebook
        self._mv_codebook = mv_codebook
        self._quantize_mat = quantize_mat
        self._file = open(file_name, 'wb')

        # padding
        self._W = ceil(self._W / self._block_size)
        self._H = ceil(self._H / self._block_size)

        # TODO: check parameter is legal

        self._time_stamp = 0
        self._last_reconstruct_img = np.zeros((self._H, self._W), dtype=np.uint8)

    def encode(self, img, mask=None):
        """encode function, give a frame `img`, return it's bitstream
        Args:
            img: image frame, a numpy array with shape (W, H)
            mask: human face & body mask, unimplenmented
        """
        if mask is not None:
            raise NotImplementedError
        img = self.preprocessing(img)
        # motion estimation
        res, mv = self.motion_estimation(img)
        res_blocks = img2blocks(res, self._block_size)
        mv_image_shape = (self._H / self._block_size, self._W / self._block_size)
        mv_imgx, mv_imgy = mvlist2img(mv, mv_image_shape)
        # dct transform
        dct_blocks = DCT_transform(res_blocks, kernel_size=8)
        # quantize
        q_blocks = quantizer(dct_blocks, self._quantize_mat)
        # zigzag scan and RLE coding
        # TODO: not implenment
        # entropy coding
        res_stream = to_entropy_code(q_blocks, self._block_codebook)
        mvx_stream = to_entropy_code([mv_imgx], self._mv_codebook[0]) 
        mvy_stream = to_entropy_code([mv_imgy], self._mv_codebook[1])

        # save reconstruct frame
        re_dct_blocks = dequantizer(q_blocks, self._quantize_mat)
        re_res_blocks = reverse_DCT_transform(re_dct_blocks)
        image_shape = (self._H, self._W)
        re_res = blocks2img(re_res_blocks, self._block_size, image_shape)
        re_image = self.motion_reconstruct(re_res, mv)
        self._last_reconstruct_img = re_image

        # update time stamp
        self._time_stamp += 1
        self.to_bitstream()
    def preprocessing(self, img):
        """ pre-process image frame
        """
        # NOTE: current version do nothing
        return img

    def motion_estimation(self, img):
        """ motion estimation, return residue and motion vector(a list)
        Args:
            img: image frame, a numpy array with shape (W, H)
        Returns:
            (res, mv): residue image and motion vector
        """
        res = img
        mv = [(0, 0) for _ in range(self._H * self._W // (self._block_size ** 2))]
        return res, mv
    
    def motion_reconstruct(self, res, mvlist):
        """ reconstruct image
        Args:
            res: residue array
            mvlist: motion vector list
        Returns:
            re_image: reconstruct image
        """
        return res
    
    def to_bitstream(self, res_stream, mvx_stream, mvy_stream):
        """ get current frame's bitstream
        """
        def write_str(_str, len_str):
            assert(len_str % 8 == 0)
            for i in range(len_str // 8):
                byte = '0b' + _str[8 * i: 8 * (i + 1)]
                byte = int(byte, 2)
                stream.write(byte, np.uint8)

        # calculate motion vector and residue length
        mvx_len = len(mvx_stream)
        mvy_len = len(mvy_stream)
        res_len = len(res_stream)

        # build stream
        stream = BitStream()
        # head
        stream.write(self._time_stamp, np.uint16)
        if self._time_stamp == 0:
            stream.write(self._H, np.uint16)
            stream.write(self._W, np.uint16)
            # TODO: adaptive quantization matrix
            stream.write(64*[0],np.uint8)
        stream.write(mvx_len, np.uint16)
        stream.write(mvy_len, np.uint16)
        stream.write(res_len, np.uint32)
        # body
        write_str(mvx_stream, mvx_len)
        write_str(mvx_stream, mvy_len)
        write_str(res_stream, res_len)
        
        # write
        self._file.write(stream.read(bytes))
    
    def finished(self):
        """ close file
        """
        self._file.close()

if __name__ == "__main__":
    pass
