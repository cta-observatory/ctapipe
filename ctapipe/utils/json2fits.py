from astropy.io import fits
from traitlets.config.loader import Config
from json import load
import logging
import warnings

logger = logging.getLogger(__name__)

__all__ = ['traitlets_config_to_fits', 'json_to_fits']


def traitlets_config_to_fits(config, fits_filename, overwrite=True):
    """Write a FITS file that represents configuration.

    Parameters
    ----------
    config : traitlets.config.loader.Config
        a traitlets.config.loader.Config to write in FITS format
    fits_filename : str
        FITS file name to write
    overwrite : bool
        When True, overwrite the output file if exists.

    Raises
    ------
    OSError : If FITS file containing the traitlets config is not written
    """

    if not isinstance(config, Config):
        raise TypeError('Config must be an instance of traitlets.config.loader.Config')

    # hduList will contain one TableHDU per section
    hdu_list = fits.HDUList()
    # get all Configuration entries
    # loop over section
    for section, entry in config.items():
        header = fits.Header()
        for key, value in entry.items():

            # CONTINUE and HIERARCH are incompatible, so we have to decide
            # to either truncate key or value. I went for the key. @maxnoe
            if isinstance(value, str):
                if len(key) > 8 and (len(key) + len(value)) > 70:
                    warnings.warn(
                        'Key "{}" will be truncated to {}'.format(key, key[:8])
                    )
                    key = key[:8]

            header[key] = value

        table_0 = fits.TableHDU(data=None, header=header, name=section)
        hdu_list.append(table_0)
    try:
        hdu_list.writeto(fits_filename, overwrite=overwrite)
    except OSError:
        logging.exception(f'Could not do save {fits_filename}')
        raise


def json_to_fits(json_filename, fits_filename, overwrite=True):
    """Write a FITS file that represents json file for traitlets configuration.

    Parameters
    ----------
    json_filename : str
        json file name containing traitlets Configuration.
        Only one level of section is allowed.
    fits_filename : str
        FITS file name to write
    overwrite : bool
        When True, overwrite the output file if exists.

    Raises
    ------
    OSError : if FITS file containing a copy of json content is not written
    FileNotFoundError : if fits_filename could not be open
    """
    try:
        f = open(json_filename, 'r')
        # hduList will contain one TableHDU per section
        hduList = fits.HDUList()
        json_object = load(f)
        # Create a global header for key/value that not depend of a section
        global_header = fits.Header()

        # loop over json entries (corresponding to Python class or general purpose)
        for section, entry in json_object.items():
            header = fits.Header()

            if isinstance(entry, dict):
                for key, value in entry.items():
                    if isinstance(value, dict):
                        raise ValueError(
                            'Malformed json file: fits header cannot contain subobjects'
                        )

                    # CONTINUE and HIERARCH are incompatible, so we have to
                    # decide to either truncate key or value.
                    # I went for the key. @maxnoe
                    if isinstance(value, str):
                        if len(key) > 8 and (len(key) + len(value)) > 70:
                            warnings.warn(
                                'Key "{}" will be truncated to {}'.format(key, key[:8])
                            )
                            key = key[:8]

                    header[key] = value

                # create a new TableHDU with current header and append it to hduList
                table_0 = fits.TableHDU(data=None, header=header, name=section)
                hduList.append(table_0)

            else:
                # entry is not a dictionary of key value but a simple value.
                # That means last section is not a section but a general purpose entry
                global_header[section] = entry

        # create a new TableHDU for global_header and append it to hduList
        table_0 = fits.TableHDU(data=None, header=global_header, name='GLOBAL')
        hduList.append(table_0)
        # write hduList to FITS file
        try:
            hduList.writeto(fits_filename, overwrite=overwrite)
        except OSError:
            logging.exception(f'Could not do save {fits_filename}')
            raise

    except FileNotFoundError:
        logging.exception(f'Could not open  {fits_filename}')
        raise
