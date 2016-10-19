from abc import abstractmethod
from .component import Component


class Processor(Component):
    '''
    A processor is a traitlets configurable class that acts like a function

    This should be the baseclass for everything that would normally be just a function
    if it would not need a log and the configurability.
    '''

    @abstractmethod
    def __call__(self):
        '''
        The __call__ method must be overriden by every base class to implement
        what the Processor is meant to do
        '''
        pass
