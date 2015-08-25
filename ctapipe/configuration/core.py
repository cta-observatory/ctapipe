# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Configuration service utilities
"""

from argparse import ArgumentParser
from configparser import RawConfigParser
from astropy.io import fits

import numpy as np
import sys

__all__ = ['ConfigurationException', 'Configuration']


class ConfigurationException(Exception):

    def __init__(self, msg):
        self.msg = msg


class Configuration(ArgumentParser):
    DEFAULT = "DEFAULT"
    FITS = "FITS"
    INI = "INI"
    DATAIMPL = "DATAIMPL"
    HEADERIMPL = "HEADERIMPL"
    VALUE_INDEX = 0
    COMMENT_INDEX = 1

    def __init__(self, other=None):
        """Configuration initialization if other is not None and its type is
        Configuration, its _entries dictionary content is copy to this
        Configuration class

        Parameters
        ----------
        other : Configuration object, optional
            Copies other._entries dictionary's item in self._entries

        """
        ArgumentParser.__init__(self)

        # Copy other dictionary entries to self._entries
        if other is not None and isinstance(other, Configuration):
            self._entries = dict(other._entries)
        else:
            self._entries = dict()  # 1 SECTION -> n KEY -> pair(value,comment)

    def parse_args(self, args=None, namespace=None):
        """Convert argument strings to objects and assign them as an entry in
        self.__dict__['DEFAULT'] Previous calls to add_argument()
        determine exactly what objects are created and how they are
        assigned.  See the documentation for add_argument() for
        details.  By default, the argument strings are taken from
        sys.argv

        Parameters
        ----------
        args : object , optional
        The argument strings are taken from args instead of sys.args
        Warning: args must contains only arguments(flag/value) no program name

        namespace: object , optional
        Populates special attribute __dict__ containing
        the namespace’s symbol table

        Returns:
        --------
        Dictionary containing configuration entries for 'DEFAULT' section

        """
        # Read arguments from sys.argv and return those previously
        # added result = super(Configuration,
        # self).parse_args(args,namespace=namespace)
        result = super().parse_args(args, namespace=namespace)
        args = vars(result)
        
        # Add arguments(key, value) for DEFAULT section
        for key, value in args.items():
            self.add(key, value, "From command line arguments", self.DEFAULT)
            self.__dict__[key] = value

        return self._entries[self.DEFAULT]

    def add(self, key, value, comment="", section=DEFAULT):
        """
        Create section if not already exist in self._entries 
        Add configuration variable for corresponding section/key
        Into 'DEFAULT' section by default

        Parameters:
        -----------
        key : str
            key for the new entry to be add
        value : str
            value for the new entry to be add
        comment : str , optional
            comment for the new entry to be add
        section : str , optional
            section for the new entry to be add

        Returns:
        --------
        True is option is added
        False is option already exist
        """
        if section not in self._entries:
            self._entries[section] = dict()
        if key not in self._entries[section]:
            self._entries[section][key] = (value, comment)
            return True
        else:
            print(section, key, "already exist", file=sys.stderr)
            return False

    def has_key(self, key, section=DEFAULT):
        """
        Checks if a configuration entry exist

        Parameters:
        -----------
        key: str
            key to search in section
        section: str , optional
            section to search key ('DEFAULT' section is used by default)
        Returns:
        --------
        whether the given option exists in the given section or not.
        """
        if section in self._entries:
            return key in self._entries[section]  # return True is key exists
        return False

    def get(self, key, section=DEFAULT):
        """
        Get a configuration entry value

        Parameters:
        -----------
        key: str
            key to search in section
        section: str , optional
            section to search key ('DEFAULT' section is used by default)

        Returns:
        --------
        value for corresponding section/key pair
        None is no suitable value exists for section/key
        """
        if not self.has_key(key, section):
            return None
        return self._entries[section][key][self.VALUE_INDEX]

    def get_comment(self, key, section=DEFAULT):
        """
        get a configuration entry comment

        Parameters:
        -----------
        key: str
            key to search in section 
        section: str , optional
            section to search key ('DEFAULT' section is used by default)

        Returns
        ------- 
        comment for corresponding section/key pair
        None is no suitable value exits for section/key
        """
        if not self.has_key(key, section):
            return None
        return self._entries[section][key][self.COMMENT_INDEX]

    def write(self, filename, impl=FITS, implementation=DATAIMPL):
        """
        write configuration entries to a file.

        Parameters:
        -----------
        filename: str
            Full path name:  Save all configuration entries
            to this given filename
        impl: str , optional
        "FITS" -> use Fits format
        "INI"  -> use windows style ini format
        """

        if (impl == self.FITS):
            self._write_fits(filename, implementation)

        # Write an .ini-format representation of the configuration state.
        elif (impl == self.INI):
            config_parser = RawConfigParser()
            self._fill(config_parser)
            with open(filename, 'w') as config_file:
                config_parser.write(config_file)
        else:
            print("Format:", impl, 'not allowed', file=sys.stderr)

    def read(self, filenames, impl=FITS, implementation=DATAIMPL, encoding=None):
        """
        Read filename or a list of filenames and parse configuration entries.

        Files that cannot be opened are silently ignored; this is
        designed so that you can specify a list of potential
        configuration file locations (e.g. current directory, user's
        home directory, system wide directory), and all existing
        configuration files in the list will be read.  A single
        filename may also be given.

        Parameters:
        -----------
        filename: str
            Full path name or list of full path name containing configuration
            entries to parse
        impl: str , optional
             FITS -> use Fits format
             INI  -> use windows style ini format
        implementation: str , optional
            DATAIMPL -> Use Fits data table
            HEADERIMP -> Use fits header

        Returns:
        ------- 
        list of successfully read files.
        """
        if impl == self.INI:
            config_parser = RawConfigParser()

            config_parser.optionxform = lambda option: option
            success_list = config_parser.read(filenames, encoding)
            self._addOptionFromParser(config_parser)
            return success_list

        elif impl == self.FITS:
            return self._read_fits(filenames, implementation, encoding)
        else:
            print("Format:", impl, 'not allowed', file=sys.stderr)
            return list()

    def list(self, file=sys.stdout, flush=False):
        """ 
        print all options (DEFAULT included)
        Parameters:
        ----------
        file: 
            file objects used by the interpreter for standard  output 
            and errors:
        flush: boolean , optional
            flush keyword argument is true, the stream is forcibly flushed.
        """
        for section in self._entries.keys():
            print("[", section, "]", file=file, flush=flush)
            for key, value_comment in self._entries[section].items():
                print (key, "=", value_comment[self.VALUE_INDEX], "; ",
                       value_comment[self.COMMENT_INDEX], file=file,
                       flush=flush)

    def _fill(self, config_parser):
        """
        Fills a Config_parser object with self._entries
        """
        if not isinstance(config_parser, RawConfigParser):
            return None

        # set RawConfigParser ro case sensitive
        config_parser.optionxform = lambda option: option
        sections = self._entries.keys()
        for section in sections:
            # dico[section]={}
            if not section == self.DEFAULT:
                config_parser.add_section(section)
            for key, value_comment_tuple in self._entries[section].items():
                # dico[section][key] = value
                value_comment = value_comment_tuple[
                    self.VALUE_INDEX] + " ; " + value_comment_tuple[self.COMMENT_INDEX]
                config_parser.set(section, key, value_comment)
        return config_parser

    def _write_fits(self, filename, implementation=DATAIMPL):
        """Write an FITS file representation of the configuration state.
        """
        if implementation != self.DATAIMPL and implementation != self.HEADERIMPL:
            print("Implementation :", implementation,
                  'not allowed', file=sys.stderr)
            return
        # hduList will contain one TableHDU per section
        hduList = fits.HDUList()

        # get all Configuration entries
        # loop over section
        for section in self._entries.keys():
            if implementation == self.DATAIMPL:
                # prepare 3 array
                key_array = []
                value_array = []
                comment_array = []
                # loop over section entries and fill arrays
                for key, value_comment in self._entries[section].items():
                    key_array.append(key)
                    value_array.append(value_comment[self.VALUE_INDEX])
                    comment_array.append(value_comment[self.COMMENT_INDEX])
                # create fits.Column form filled arrays
                ckey = fits.Column(
                    name='key', format='A256', array=np.array(key_array))
                cvalue = fits.Column(
                    name='value', format='A256', array=np.array(value_array))
                ccomment = fits.Column(
                    name='comments', format='A256',
                    array=np.array(comment_array))
                # Create the table
                hdu = fits.TableHDU.from_columns([ckey, cvalue, ccomment])
                hdu.name = section
                # append table to hduList
                hduList.append(hdu)

            elif (implementation == self.HEADERIMPL):
                header = fits.Header()
                for key, value_comments in self._entries[section].items():
                    header[key] = (
                        value_comments[self.VALUE_INDEX],
                        value_comments[self.COMMENT_INDEX])

                table_0 = fits.TableHDU(data=None, header=header, name=section)
                hduList.append(table_0)

        hduList.writeto(filename, clobber=True)

    def _read_fits(self, filenames, implementation=DATAIMPL, encoding=None):
        """Read and parse a Fits file or a list of Fits files.
            Return list of successfully read files.
        """
        if isinstance(filenames, str):
            filenames = [filenames]

        read_ok = []

        if implementation == self.DATAIMPL:
            for filename in filenames:
                hdulist = fits.open(filename)
                for hdu in hdulist:
                    section = hdu.name
                    data = hdu.data
                    if not data is None:
                        for (key, value, comment) in data:
                            try:
                                self.add(key, value, comment=comment,
                                         section=section)
                            except ConfigurationException as e:
                                print(e, file=sys.stderr)
                            except:
                                pass
                read_ok.append(filename)

        elif (implementation == self.HEADERIMPL):
            for filename in filenames:
                hdulist = fits.open(filename)
                for hdu in hdulist:
                    header = hdu.header
                    section = hdu.name
                    for key in header:
                        try:
                            self.add(key, header[key],
                                     section=section,
                                     comment=header.comments[key])
                        except ConfigurationException as e:
                            print(e, file=sys.stderr)
                        except:
                            pass
                read_ok.append(filename)
                read_ok.append(filename)

        else:
            print("Implementation :", implementation,
                  'not allowed', file=sys.stderr)

        return read_ok

    def _addOptionFromParser(self, config_parser):
        """
            fill self._entries from a RawConfigParser
        """
        if not isinstance(config_parser, RawConfigParser):
            return False

        for section in config_parser.sections():
            for key, value_comment in config_parser.items(section):
                foo = value_comment.split(" ;")
                value = foo[self.VALUE_INDEX]
                comment = foo[self.COMMENT_INDEX]
                comment = comment[1:]
                self.add(key, value, comment=comment, section=section)

        for key, value_comment in config_parser.defaults().items():
            foo = value_comment.split(" ;")
            value = foo[self.VALUE_INDEX]
            comment = foo[self.COMMENT_INDEX]
            comment = comment[1:]
            self.add(key, value, comment=comment, section=self.DEFAULT)
        # add default section
        return True
