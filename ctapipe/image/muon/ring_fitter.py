
class RingFitter(Component):
    """This is the base class from which all ring fitters should inherit from"""

    def __init__(self):
        pass

    def fit(x,y,weight,times=None):
        """overwrite this method with your favourite ring fitting algorithm
        inputs:
        x      = vector of pixel x-coordinates as astropy quantities
        y      = vector of pixel y-coordinates as astropy quantities
        weight = vector of pixel weights 
        times  = optional vector of pixel DAQ times as astropy quantities"""
        return None
