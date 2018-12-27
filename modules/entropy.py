
from .error import *
def to_entropy_code(blocks, codebook):
    """ transform blocks to bit string
    Args:
        blocks: a list
        codebook: a dictionary
    Returns:
        stream: a string including '0' or '1', length should be multiple of 8
    """
    stream = ''
    for block in blocks:
        block = block.flatten()
        for item in block:
            try:
                stream += codebook[item]
            except KeyError:
                raise EntropycodeError
    return stream


def from_entropy_code(stream, codebook, block_length):
    """ transform stream to blocks
    """
    r_codebook = {v:k for k,v in codebook.items()}
    blocks = []
    # TODO: entropy decode
    raise NotImplementedError
