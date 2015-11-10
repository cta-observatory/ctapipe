from astropy.table import Table
from astropy.io import fits
import numpy as np
from argparse import ArgumentParser
import sys
import fitsio

import wx
try:
    import wx.lib.wxcairo
    import cairo
    haveCairo = True
except ImportError:
    haveCairo = False

class Poly(object):
    """ a drawable polygon  with N vertices"""
    def __init__(self, order=6,theta0=0.0 ):
        """
        """
        angles = (np.linspace(0.0,360.0,order+1) + theta0) * np.pi/180.0
        self.vertices =  np.array(list(zip(np.cos(angles),np.sin(angles))))

    def draw(self, cr, xx, yy, size=1.0):
        """
        Draw a polygon
        Arguments:
        - `cr`: cairo context
        - `xx`: center position
        - `yy`: center position
        - `size`: size
        """

        cr.move_to(xx,yy) #center of hexagon
        cr.rel_move_to(0.5*size,-1.0*size)

        for vx,vy in self.vertices:
            cr.rel_line_to( vx*size,vy*size )

        cr.close_path()


def cairo_draw_camera( cr, pix_x, pix_y, pix_r, pix_val, poly=Poly(6,theta0=90)):
    """
    Arguments:
    cr --  Cairo context
    pix_x -- array-like list of pixel X positions
    pix_y -- array-like list of pixel Y positions
    pix_r -- array-like list of pixel radii (or size)
    pix_val -- array-like list of pixel intensities
    poly -- a Poly instance to draw for each pixel
    width -- screen width
    height -- screen height
    aspect -- zoom level
    """

    pix_val[pix_val>1.0] = 1.0
    pix_val[pix_val<0.0] = 0.0

    for xx,yy,rr,vv in zip(pix_x,pix_y,pix_r, pix_val):
        cr.set_source_rgb(1, 0, 0)
        poly.draw(cr,xx,yy,rr)
        cr.set_source_rgb(vv**0.2, vv**0.7, vv**2.0)
        cr.fill()


class CameraFrame(wx.Frame):
    def __init__(self, parent, title,numcams=1):
        wx.Frame.__init__(self, parent, title=title, size=(640,640))
        self.canvas = []
        self.canvas = CameraPanel(self)
        self.Show()

class CameraPanel(wx.Panel):
    """ a Cairo Panel that draws a Cherenkov Camera with Hexagonal Bins """
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, style=wx.BORDER_SIMPLE)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.pix_x = np.array([0])
        self.pix_y = np.array([0])
        self.pix_r = np.array([0])
        self.pix_val = np.array([0])
        self._scale = 1.2
        self._rotation=0.0
        self.refreshtimer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self.refreshtimer)
        #self.refreshtimer.Start(1)


    def _on_timer(self,event ):
        self.Refresh()

    def set_camera_geom( self, pix_x, pix_y, pix_r, poly=Poly(6,theta0=90) ):
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.pix_r = pix_r

    def set_data( self, pix_val ):
        self.pix_val = pix_val

    def set_zoom(self, val):
        self._scale = val

    def set_rotation(self,val):
        self._rotation = val

    def _on_paint(self, evt):
        dc = wx.PaintDC(self)
        width, height = self.GetClientSize()
        cr = wx.lib.wxcairo.ContextFromDC(dc)

        # background color:
        cr.set_source_rgb(0.2, 0.4, 0.6)
        cr.rectangle(0, 0, width, height)
        cr.fill()


        cr.translate(width/2.,height/2.) # move to center
        cr.scale( width/2.0*self._scale,height/2.0*self._scale )
        cr.rotate( self._rotation )
        cr.set_line_width(0.005)

        cairo_draw_camera(cr, self.pix_x, self.pix_y,
                          self.pix_r, self.pix_val)




if __name__ == '__main__':

    def update_data(wxevent):
        global pix_val,eventnum, tevents
        global camframe, refreshtimer,max_events,tel_id

        event = tevents[eventnum]

        compressed_val= event["TEL_IMG_INT"]
        compressed_ipix = event["TEL_IMG_IPIX"]

        # apply scaling and cuts:
        compressed_val /= 0.90*compressed_val.max()
        compressed_val[compressed_val>1.0] = 1.0
        compressed_val[compressed_val<0.0] = 0.0
        # decompress the image:
        pix_val[:] = 0 # np.random.uniform(0.0,0.5, size=len(pix_val))
        pix_val[compressed_ipix]  = compressed_val

        camframe.canvas.set_data(pix_val)
        camframe.canvas.set_zoom( 1.0 + 1.0*np.sin(0.01*eventnum)**2 )
        camframe.canvas.set_rotation(0.005*eventnum )
        camframe.canvas.Refresh()
        eventnum += 1
        if eventnum >= max_events:
            eventnum = 0


    parser = ArgumentParser( prog="camdisplay" )
    parser.add_argument("--camfile",type=str, default="chercam.fits.gz")
    parser.add_argument("televentsfile",type=str)
    opts = parser.parse_args(sys.argv[1:])

    # load the camera info:
    print("LOADING CAMERA from {}".format(opts.camfile))
    camera = Table.read("chercam.fits.gz")
    mask = camera['CAM_ID'] == 1
    pix_x = camera['PIX_POSX'][mask]
    pix_y = camera['PIX_POSY'][mask]
    pix_r = camera['PIX_DIAM'][mask]/2.0
    pix_val = np.zeros(shape=len(pix_x))+np.random.normal(size=len(pix_x))


    # load the data and do some stuff with it:
    print("LOADING DATA from {}".format(opts.televentsfile))
    header = fitsio.read_header(opts.televentsfile, ext="TEVENTS")
    max_events = header["NAXIS2"]
    tevents = fitsio.FITS(opts.televentsfile,iter_row_buffer=1000)['TEVENTS']
    print(" --> found {}  events".format(max_events))
    eventnum = 0


    if haveCairo:
        app = wx.App(False)
        camframe = CameraFrame(None, 'Basic Camera Display')
        camframe.canvas.set_camera_geom( pix_x, pix_y, pix_r )
        camframe.canvas.set_data( pix_val )

        refreshtimer = wx.Timer(camframe)
        camframe.Bind(wx.EVT_TIMER, update_data, refreshtimer)
        refreshtimer.Start(10)

        app.MainLoop()
    else:
        print("Error! PyCairo or a related dependency was not found")
