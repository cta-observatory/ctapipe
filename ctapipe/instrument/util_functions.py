import hessio as h
from astropy.io import fits

__all__ = ['get_file_type','load','close','LoadClose']
           

def get_file_type(filename):
    """
    Returns a string with the type of the given file (guessed from the
    extension). The '.gz' or '.bz2' compression extensions are
    ignored.

    >>> get_file_type('myfile.fits.gz')
    'fits'

    """
    if  ('fit' or 'FITS' or 'FIT') in filename:
        return 'fits'
    else:
        return 'hessio'

def load(filename):
    """
    calls the specific load function depending on the file
    extension of the given file.
    
    Parameters
    ----------
    filename: string
        name of the file
    """
    ext = get_file_type(filename)
    function = getattr(LoadClose,"load_%s" % ext)
    return function(filename)
        
    #if 'simtel.gz' in filename:
    #    event = load_hessio(filename)
    #    nextevent_hessio()
    #    return event
    #elif 'fits' in filename:
    #    return load_fits(filename)

def close(filename,item):
        """
        calls the specific close function depending on the file
        extension of the given file.
        
        Parameters
        ----------
        filename: string
            name of the file
        item: of various type depending on the file extension
            return value of the opening/loading process of the file
        """
        
        ext = get_file_type(filename)
        function = getattr(LoadClose,"close_%s" % ext)
        return function(item)      
        
        #if 'simtel.gz' in filename:
        #    close_hessio()
        #elif 'fits' in filename:
        #    close_fits(item)
        
        
class LoadClose:
    
    """`LoadClose` is a class containing the load and close functions for
    the different file extensions"""    
    
    def load_hessio(filename):
        """
        Function to open and load a hessio file
        
        Parameters
        ----------
        filename: string
            name of the file
        """
        event = h.file_open(filename)
        LoadClose.nextevent_hessio()
        print("Hessio file %s has been opened" % filename)
        return event
        
    def nextevent_hessio():
        """
        Function to switch to the next event within the open hessio file
        """
        next(h.move_to_next_event())
        
    def close_hessio(item):
        """Function to close a hessio file"""
        h.close_file()
        print("Hessio file has been closed.")
    
    def load_fits(filename):
        """
        Function to open and load a fits file
        
        Parameters
        ----------
        filename: string
            name of the file
        """
        hdulist = fits.open(filename)
        print("Fits file %s has been opened" % filename)
        return hdulist
    
    def close_fits(hdulist):
        """
        Function to close a fits file
        
        Parameter
        ---------
        hdulist: HDUList
            HDUList object of the fits file
        """
        print("Fits file has been closed.")    
        hdulist.close()
    
    def load_ascii(filename):
        """Function to open and load an ASCII file"""
        print("ASCI file %s has been opened" % filename)
        
    def close_ascii():
        """Function to open and load an ASCII file"""
        print("ASCII file has been closed")
