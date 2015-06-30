from ctapipe import io
import matplotlib.pylab as plt

if __name__ == '__main__':

    layout = io.get_array_layout("hess")
    X = layout['POSX']
    Y = layout['POSY']

    plt.scatter(X,Y,s=50)
    plt.title("Array Layout: {}".format(layout.meta['ARRAY']))
    plt.xlabel("{} ({})".format(X.name, X.unit))
    plt.ylabel("{} ({})".format(Y.name, Y.unit))

    for tel in layout:
        name = "CT{tid}:{tclass}".format(tid=tel['TELID'],
                                         tclass=io.tel_class_name(tel['CLASS']))
        plt.text(tel['POSX'],tel['POSY'], name)


    plt.show()
