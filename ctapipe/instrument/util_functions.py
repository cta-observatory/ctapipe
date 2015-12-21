import pyhessio as h
import numpy as np
import imp
import os
from astropy.io import fits
import random
import textwrap

__all__ = ['get_file_type','load_hessio','nextevent_hessio','close_hessio',
           'load_fits','close_fits','get_var_from_file','load_ascii',
           'close_ascii']
           

def get_file_type(filename):
    """
    Returns a string with the type of the given file (guessed from the
    extension). The '.gz' or '.bz2' compression extensions are
    ignored.

    >>> get_file_type('myfile.fits.gz')
    'fits'

    """
    if  ('fit' or 'FITS' or 'FIT') in filename or filename == 'fake_data':
        return 'fits'
    elif 'cfg' in filename:
        return 'ascii'
    else:
        return 'hessio'
    
def load_hessio(filename):
    """
    Function to open and load a hessio file
    
    Parameters
    ----------
    filename: string
        name of the file
    """
    event = h.file_open(filename)
    nextevent_hessio()
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
    if 'fake_data' in filename:
        hdulist = load_fakedata()
    else:
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
    hdulist.close()
    print("Fits file has been closed.")

def get_var_from_file(filename):
    """
    Function to load and initialize a module implemented as a Python source
    file called `filename` and to return its module objects.
    
    Parameter
    ---------
    filename: ASCII file
        file in which the module objects are defined
    """
    f = open(filename)
    global data
    data = imp.load_source('data', '', f)
    f.close()
    
def load_ascii(filename):
    """Function to open and load an ASCII file"""
    file = open(filename,'r')
    print("ASCI file %s has been opened." % filename)
    temp_filename = '%s_temp.txt' % os.path.splitext(os.path.basename(filename))[0]
    print("Temporary file ",temp_filename," created.")
    temp_file = open(temp_filename, 'w')    
    for line in file:
        if 'echo' in line:
            line = line.replace('echo','#')
        if "%" in line:
            line = line.replace('%','#')
        if line.startswith('#'):
            pass
        else:
            if '=' in line:
                index1 = line.index('=')
                if '#' in line:
                    index2 = line.index('#')-1
                else:
                    index2 = len(line)-1
                line = line[:index1+1]+'['+line[index1+1:index2]+']'+'\n'
                for i in range(4):
                    try: float(line[index1+2+i])
                    except: x = False
                    else:
                        x = True
                        break
                if x==False: line = line[:index1+2]+'"'+line[index1+2:index2+1]+'"'+']'+'\n'
            else:
                line = '\n'
        line = textwrap.dedent(line)
        temp_file.write(line) 
    file.close()
    print("ASCII file has been closed.")
    temp_file.close()
    print("Temporaryly created file has been closed.")
    get_var_from_file(temp_filename)
    os.remove(temp_filename)
    print("Temporaryly created file has been removed.")
    return data
    
    
def close_ascii(filename):
    """Function to close an ASCII file"""
    print("ASCII file has been closed.")
    

def load_fakedata():
    """
    Function writing faked date into an astropy.io.fits.hdu.table.BinTableHDU
    """
    tel_num = 10
    tel_id = [random.randrange(1,124,1) for x in range(0,tel_num)]
    tel_posX = [random.uniform(1,100) for x in range(0,tel_num)]
    tel_posY = [random.uniform(1,100) for x in range(0,tel_num)]
    tel_posZ = [random.uniform(1,100) for x in range(0,tel_num)]
    l0id = [i for i in range(0,tel_num)]
    
    mirror_area = [random.uniform(1,100) for x in range(0,5)]
    mirror_number = [random.randrange(1,124,1) for x in range(0,tel_num)]
    
    col0 = fits.Column(name='L0ID',format='I',array=l0id)
    col1 = fits.Column(name='TelID',format='I',array=tel_id)
    col2 = fits.Column(name='TelX',format='D',array=tel_posX)
    col3 = fits.Column(name='TelY',format='D',array=tel_posY)
    col4 = fits.Column(name='TelZ',format='D',array=tel_posZ)
    col5 = fits.Column(name='NMirrors',format='I',array=mirror_number)
    col6 = fits.Column(name='MirrorArea',format='D',array=mirror_area)
    
    cols1 = fits.ColDefs([col0,col1,col2,col3,col4,col5,col6])
    tbhdu1 = fits.BinTableHDU.from_columns(cols1)
    
    pixel_num = 1140    
    l1id = [i for i in range(0,tel_num*pixel_num)]    
    pixel_posX = [random.uniform(1,100) for x in range(0,tel_num*pixel_num)]
    pixel_posY = [random.uniform(1,100) for x in range(0,tel_num*pixel_num)]
    l0id_prime = np.array([[l0id[i] for k in range(0,pixel_num)] for i in range(0,tel_num)])
    l0id_prime = l0id_prime.flatten()
    pixel_id = np.array([[i for i in range(1,pixel_num+1)] for j in range(0,tel_num)])
    pixel_id = pixel_id.flatten()
    
    col01 = fits.Column(name='L1ID',format='I',array=l1id)
    col11 = fits.Column(name='L0ID',format='I',array=l0id_prime)
    col21 = fits.Column(name='PixelID',format='I',array=pixel_id)
    col31 = fits.Column(name='XTubeMM',format='D',array=pixel_posX)
    col41 = fits.Column(name='YTubeMM',format='D',array=pixel_posY)
    
    cols2 = fits.ColDefs([col01,col11,col21,col31,col41])
    tbhdu2 = fits.BinTableHDU.from_columns(cols2)
    
    hdulist = fits.HDUList([tbhdu1,tbhdu2])
    
    return hdulist
    