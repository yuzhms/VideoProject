class Error(Exception):
    """ Base-class for all exceptions raised by application
    """
    pass
    
class ShapeError(Error):
    """ shape inappropriate error
    """
    pass

class EntropycodeError(Error):
    """ can't code/decode to/from entropy code
    """
    pass