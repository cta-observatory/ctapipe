from astropy.io import fits
from traitlets.config.loader import Config
from json import load
import logging

logger = logging.getLogger(__name__)

__all__ = ['traitlets_config_to_fits', 'json_to_fits']

def traitlets_config_to_fits(config, fits_filename, clobber=True):
    """Write a FITS file that represents configuration.
    Parameters
    ----------
    config : traitlets.config.loader.Config
        a traitlets.config.loader.Config to write in FITS format
    fits_filename : str
        FITS file name to write
    clobber : bool
        When True, overwrite the output file if exists.
    Returns
    -------
    True is FITS file containing the traitlets config is written
    Otherwise False
    """

    if not isinstance(config,Config):
        logger.error("traitlets_config_to_fits: config must be an instance of traitlets.config.loader.Config")
        return False
    # hduList will contain one TableHDU per section
    hduList = fits.HDUList()
    # get all Configuration entries
    # loop over section
    for section,entry in config.items():
        header = fits.Header()
        for key,value in entry.items():
            header[key] = value
        table_0 = fits.TableHDU(data=None, header=header, name=section)
        hduList.append(table_0)
    try:
        hduList.writeto(fits_filename, clobber=clobber)
    except OSError as e:
        logger.error(str(e))
        return False
    return True




def json_to_fits(json_filename, fits_filename, clobber=True):
    """Write a FITS file that represents json file for traitlets configuration.
    Parameters
    ----------
    json_filename : str
        json file name containing traitlets Configuration.
        Only one level of section is allowed.
    fits_filename : str
        FITS file name to write
    clobber : bool
        When True, overwrite the output file if exists.
    Returns
    -------
    True is FITS file containing a copy of json content is written
    Otherwise False
    """
    try:
        f = open(json_filename, 'r')
        # hduList will contain one TableHDU per section
        hduList = fits.HDUList()
        json_object = load(f)
        # Create a global header for key/value that not depend of a section
        global_header = fits.Header()
        #loop over json entries (corresponding to Python class or general purpose)
        for section,entry in json_object.items():
            #create a new header for this section
            header = fits.Header()
            if  isinstance(entry, dict):
                #loop over key/value entries
                for k, v in entry.items():
                    if isinstance(v, dict):
                        logger.error('Fits Header cannot contains subheaders.')
                        logger.error('Please correct your json file to follow requirements.')
                        return False
                    header[k] = v
                # create a new TableHDU with current header and append it to hduList
                table_0 = fits.TableHDU(data=None, header=header, name=section)
                hduList.append(table_0)
            else:
                # entry is not a dictionary of key value but a simple value.
                # That means last section is not a section but a general purpose entry
                global_header[section] = entry

        # create a new TableHDU for global_header and append it to hduList
        table_0 = fits.TableHDU(data=None, header=global_header, name="GLOBAL")
        hduList.append(table_0)
        # write hduList to FITS file
        try:
            hduList.writeto(fits_filename, clobber=clobber)
        except OSError as e:
            logger.error(str(e))
            return False
        return True
    except FileNotFoundError:
        logger.error('No such file or directory:\''+json_filename+'\'')
        return False
