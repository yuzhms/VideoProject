from math import ceil
from queue import Queue

import numpy as np
from bitstream import BitStream

from ..modules.DCT import DCT_transform, reverse_DCT_transform
from ..modules.ustils import img2blocks, blocks2img, mvlist2img, img2mvlist
from ..modules.quantizer import quantizer, dequantizer
from ..modules.entropy import to_entropy_code, from_entropy_code
from ..modules.error import *

class BaseDecoder(object):
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
    def __init__(self, blcok_size, block_codebook, mv_codebook,\
                quantize_mat, file_name, bg_buffer_size):
        """
        block_size: decode block size, should be an integer like 8
        block_codebook: codebook for block entropy coding, a dictionary
        mv_codebook: codebook for mv entropy coding, a tuple of two dictionary
        quantize_mat: quantization matrix
        file_name: file name
        bg_buffer size: background buffer size
        """
        self._block_size = blcok_size
        self._block_codebook = block_codebook
        self._mv_codebook = mv_codebook
        self._quantize_mat = quantize_mat
        self._file = open(file_name, 'rb')

        self._time_stamp = 0
        self._last_reconstruct_img = np.zeros((self._H, self._W), dtype=np.uint8)

        self._H = None
        self._W = None

    def decode(self):
        """decode function
        """

        res_stream, (mvx_stream, mvy_stream) = self.from_bitstream()
        # get blocks from entropy code
        q_blocks = from_entropy_code(res_stream, self._block_codebook,\
                    block_length=self._block_size ** 2)
        mv_imgx = from_entropy_code(mvx_stream, self._mv_codebook[0],\
                    block_length=(self._W * self._H / self._block_size ** 2))[0]
        mv_imgy = from_entropy_code(mvy_stream, self._mv_codebook[1],\
                    block_length=(self._W * self._H / self._block_size ** 2))[0]

        # dequantize
        dct_blocks = dequantizer(q_blocks, self._quantize_mat)
        # reverse zigzag scan and RLE coding
        # TODO: not implenment
        # reverse dct transform
        res_blocks = reverse_DCT_transform(dct_blocks, kernel_size=8)
        # reconstruct mv and res
        mv = img2mvlist(mv_imgx, mv_imgy)
        res = blocks2img(res_blocks, self._block_size, (self._H, self._W))
        # reconstruct frame
        img = motion_reconstruct(res, mv)
        img = postprocessing(img)
        # save img
        self._last_reconstruct_img = img
        # update time stamp
        self._time_stamp += 1
        return img

    def postprocessing(self, img):
        """ post-process image frame
        """
        # NOTE: current version do nothing
        return img

    
    def motion_reconstruct(self, res, mvlist):
        """ reconstruct image
        Args:
            res: residue array
            mvlist: motion vector list
        Returns:
            re_image: reconstruct image
        """
        return res
    
    def from_bitstream(self):
        """ get current frame's bitstream
        """
        def read_str(len_str):
            assert(len_str % 8 == 0)
            _str = ''
            for i in range(len_str // 8):
                byte = stream.read(np.uint8, 1)
                byte = bin(byte)[2:]
                byte = '0' * (8 - len(byte)) + byte
                _str += byte
            return _str

        stream = BitStream()
        # read from file
        time_stamp = stream.write(self._file.read(2), bytes).read(np.uint8)
        if time_stamp != self._time_stamp:
            raise Error
        if time_stamp == 0:
            stream.write(self._file.read(68), bytes)
            self._H = stream.read(np.uint16, 1)
            self._W = stream.read(np.uint16, 1)
            # TODO: adaptive quantization matrix
            _ = stream.read(np.uint8, 64) 
        stream.write(self._file.read(8), bytes)
        mvx_len = stream.read(np.uint16, 1)
        mvy_len = stream.read(np.uint16, 1)
        res_len = stream.read(np.uint32, 1)
        body_bytes = mv_len // 8 + res_len // 8
        stream.write(self._file.read(body_bytes), bytes)
        mvx = read_str(mvx_len)
        mvy = read_str(mvx_len)
        res = read_str(res_len)
        return res, (mvx, mvy)

    
    def finished(self):
        """ close file
        """
        self._file.close()

if __name__ == "__main__":
    pass
