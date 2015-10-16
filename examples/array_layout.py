from ctapipe import io
from ctapipe.visualization import ArrayDisplay
import matplotlib.pylab as plt

if __name__ == '__main__':

    plt.style.use("ggplot")
    layout = io.get_array_layout("hess")
    X = layout['POSX']
    Y = layout['POSY']
    A = layout['MIRAREA']
    A[:] = 132

    ad = ArrayDisplay(X, Y, A, title="HESS")

    # label them
    for tel in layout:
        name = "CT{tid}:{tclass}".format(tid=tel['TELID'],
                                         tclass=io.tel_class_name(tel['CLASS']))
        plt.text(tel['POSX'], tel['POSY'], name)

    ad.axes.set_xlim(-300, 300)
    ad.axes.set_ylim(-300, 300)
    plt.show()
