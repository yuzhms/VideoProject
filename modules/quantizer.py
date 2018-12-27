import numpy as np

def quantizer(blocks, mat):
    """ quantize
    Args:
        blocks: list of numpy array with dtype float32
        mat: same shape as block
    Returns:
        q_blocks: uint16
    """
    q_blocks = []
    for block in blocks:
        q_blocks.append((block / mat).astype(np.uin16))
    return q_blocks


def dequantizer(q_blocks, mat):
    """ dequantize
    Args:
        q_blocks: list of numpy array with dtype uint16
        mat: same shape as block
    Returns:
        blocks: float32
    """
    blocks =[]
    for q_block in q_blocks:
        blocks.append((q_block * mat).astype(np.float32))
    return blocks