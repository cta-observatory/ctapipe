"""
Visualization of images in a Cherenkov Camera
"""


class CameraDisplay(object):
    """Base class to Handle the display of a single camera image, with
    support for overlaying useful information. This class only
    dhandles drawing, not interaction or GUI windowing.

    The actual drawing is performed by a backend renderer class (so
    multiple backends may be implemented for various graphics systems)

    Parameters
    ----------
    geometry : `~ctapipe.io.camera.CameraGeometry`
        Definition of the camera geometry
    renderer : {"MPL",}
        Which backend to use for rendering
    """
    def __init__(self, geometry, renderer="MPL"):
        super(CameraRenderer, self).__init__()
        self.args = args
        self.renderer = renderer

        if self.renderer == "MPL":
            self.renderer = MPLCameraRenderer(geometry)
        
    def draw_image(self, image):
        """
        Parameters
        ----------
        image: array_like
            an array of pixel values corresponding to the `CameraGeometry` 
            that was used when constructing the `CameraRenderer`
        """
        self.renderer.render_image( image )

    def overlay_hillas( self, centroid, length, width, phi):
        self.renderer.draw_ellipse( centroid, length, width, phi )


        
class CameraRenderer(object):
    """base class for low-level drawing of cameras (subclasses implement the 
    functionality).
    """
    def __init__(self, geometry):
        super(CameraRenderer, self).__init__()

    def render_image(self,image):
        pass

    def overlay_ellipse( self, centroid, length, width, phi ):
        pass

    
class MPLCameraRenderer(object):
    """ 
    Render a camera using MatPlotLib
    """
    
    def __init__(self, geometry):
        super(MPLCameraRenderer,self).__init__(geometry)
        
        
        
    def render_image(self, image):
        pass




