from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt




if __name__ == '__main__':

    plt.style.use("ggplot")
    
    N=2
    M=4
    plt.figure(figsize=(10, 6))

    camera_names = CameraGeometry.get_known_camera_names()

    for ii, name in enumerate(camera_names):
        geom = CameraGeometry.from_name(name)
        plt.subplot(N,M, ii+1)
        disp = CameraDisplay(geom)
        

    plt.tight_layout()
    plt.show()
