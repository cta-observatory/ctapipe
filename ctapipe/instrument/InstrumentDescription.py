import numpy as np
import os
import sys
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
import hessio as h
from ctapipe.io.hessio import hessio_event_source
from ctapipe.io.files import get_file_type
from ctapipe import io
from ctapipe.instrument import instrument_lists as ld
import warnings
from ctapipe.visualization import ArrayDisplay
import matplotlib.pyplot as plt

#TO DO:
# - rise warnings or put values to -1 or sth. else if desired information is not stored in the file. Currently in such a case returned array is empty. Maybe it's ok like that??
# - rise warning if one trys to get a camera or optics information of a telescope of ID x which is not given in the file


def load_hessio(filename):
    """Function to open and load hessio files"""
    event = h.file_open(filename)
    print("Hessio file %s has been opened" % filename)
    return event
    
def nextevent_hessio():
    next(h.move_to_next_event())

def load_fits(filename):
    hdulist = fits.open(filename)
    print("Fits file %s has been opened" % filename)
    return hdulist

def load_ascii(filename):
    """Function to open and load an ASCII file"""
    print("ASCI file %s has been opened" % filename)

def close_hessio():
    h.close_file()
    print("Hessio file has been closed.")

def close_fits(hdulist):
    print("Fits file has been closed.")
    hdulist.close()

def close_ascii():
    print("ASCII file has been closed")



def initialize_camera(tel_id, filename, file_closed = 1):
    if 'simtel.gz' in filename:
        if file_closed:
            ld.clear_lists_camera()
            load_hessio(filename)
            nextevent_hessio()
        else:
            pass
    
        px = h.get_pixel_position(tel_id)[0]
        py = h.get_pixel_position(tel_id)[1]
        ld.pixel_posX.append(px)
        ld.pixel_posY.append(py)
        ld.camera_class.append(io.guess_camera_geometry(px*u.m,py*u.m).cam_id)
        
        #to use this, one has to go through every event of the run...
        n_channel = h.get_num_channel(tel_id)
        ld.channel_num = n_channel
        for chan in range(n_channel):
            ld.adc_sampels.append(h.get_adc_sample(tel_id,chan).tolist())

        if file_closed:
            close_hessio()
        
    elif 'fits' in filename:
        hdulist = file_closed
        if file_closed == 1:
            ld.clear_lists_camera()
            ld.clear_lists_telescope()
            hdulist = load_fits(filename)
            teles = hdulist[1].data
            ld.telescope = teles["TelID"].tolist()
        else:
            pass
        
        index = ld.telescope_id.index(tel_id)
        ld.camera_fov.append(hdulist[1].data[index]["FOV"])
        
        pixel_id_cam = []
        index2 = np.where(hdulist[2].data['L0ID'] == index)[0]
        for i in index2:
            pixel_id_cam.append(hdulist[2].data[i]['PixelID'])
        ld.pixel_id.append(pixel_id_cam)
        
        if file_closed == 1:
            close_fits(hdulist)
    
        
def initialize_optics(tel_id, filename, file_closed = 1):
    if 'simtel.gz' in filename:
        if file_closed:
            ld.clear_lists_optics()
            load_hessio(filename)
            nextevent_hessio()
        else:
            pass
    
        ld.mirror_area.append(h.get_mirror_area(tel_id))
        ld.mirror_number.append(h.get_mirror_number(tel_id))

        if file_closed:
            close_hessio()        

    elif 'fits' in filename:
        hdulist = file_closed
        if file_closed == 1:
            ld.clear_lists_optics()
            ld.clear_lists_telescope()
            hdulist = load_fits(filename)
            teles = hdulist[1].data
            ld.telescope = teles["TelID"].tolist()  
        else:
            pass
        index = ld.telescope_id.index(tel_id)
        ld.mirror_area.append(hdulist[1].data[index]["MirrorArea"])

        if file_closed == 1:
            close_fits(hdulist)

def initialize_telescope(filename, file_closed = True):
    ld.clear_lists_telescope()
    ld.clear_lists_camera()

    if 'simtel.gz' in filename:
        if file_closed:
            file_closed = load_hessio(filename)
            nextevent_hessio()
        else:
            pass

        #ld.telescope_id = h.get_telescope_ids().tolist() #--> this function only can be used if the according python wrapper hase been added to pyhessio.c and hessio.py
        ld.telescope_id = h.get_teldata_list().tolist()
        ld.telescope_num = h.get_num_telescope()

    elif 'fits' in filename:
        if file_closed:
            hdulist = load_fits(filename)
            file_closed = hdulist
        else:
            pass
        teles = hdulist[1].data
        ld.telescope_id = teles["TelID"].tolist()
        ld.telescope_posX = teles["TelX"].tolist()
        ld.telescope_posY = teles["TelY"].tolist()
        ld.telescope_posZ = teles["TelZ"].tolist()
        ld.telescope_num = len(ld.telescope_id)

    for tel in ld.telescope_id:
        initialize_camera(tel,filename,file_closed)
        initialize_optics(tel,filename,file_closed)

    if 'simtel.gz' in filename:   
        close_hessio()
    elif 'fits' in filename:
        close_fits(hdulist)



class Subarray:
    #What should be in here?

    def __init__(self):
        self.telescope = Telescope()
    
    def plotSubArray(self):
        ad = ArrayDisplay(ld.telescope_posX, ld.telescope_posY, ld.mirror_area)
        for i in range(len(ld.telescope_id)):
            name = "CT%i" % ld.telescope_id[i]
            plt.text(ld.telescope_posX[i],ld.telescope_posY[i],name,fontsize=8)
        ad.axes.set_xlim(-1000, 1000)
        ad.axes.set_ylim(-1000, 1000)
        plt.show()

    

class Telescope:
    """`Telescope` is a class that provides and gets all the information about
    a specific telescope such as the camera's characteristics"""

    def __init__(self):
        self.optics = Optics()
        self.camera = Camera()

    #Getter Functions:

    def getTelescopeNumber(self):
        return ld.telescope_num

    def getTelescopeID(self):
        return(ld.telescope_id)

    def getTelescopePosX(self):
        if len(ld.telescope_posX) == 0:
            warnings.warn("File contains no info about TelescopePosX.")
        return(ld.telescope_posX)

    def getTelescopePosY(self):
        return(ld.telescope_posY)

    def getTelescopePosZ(self):
        return(ld.telescope_posZ)


    #Write Functions:

    #Plot Functions:

    #def plot(self, list_1, list_2=0):
    #    if list_2==0:

class Optics:
    """`Optics` is a class that provides and gets all the information about
    the optics of a specific telescope."""

    def getMirrorArea(self):
        return(ld.mirror_area)

    def getMirrorNumber(self):
        return(ld.mirror_number)

    def getOptFocLen(self):
        return ld.focal_length

class Camera:
    """`Camera` is a class that provides and gets all the information about
    the camera of a specific telescope."""

    def getCameraClass(self):
        return(ld.camera_class)

    def getCameraFOV(self):
        return(ld.camera_fov)

    def getPixelID(self):
        return(ld.pixel_id)

    def getPixelX(self):
        return(ld.pixel_posX)

    def getPixelY(self):
        return(ld.pixel_posY)

    def getPixelZ(self):
        return(ld.pixel_posZ)

    #maybe not needed...
    def getNumberChannels(self):
        return ld.channel_num

    def getADCsamples(self):
        return ld.adc_samples

class Pixel:
     """`Pixel` is a class that provides and gets all the information about
    a specific pixel of a specific camera."""

     
    
