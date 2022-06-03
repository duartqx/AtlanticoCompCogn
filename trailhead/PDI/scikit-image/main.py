from src.astronaut import Astronaut
from src.coffee import Coffee

if __name__ == '__main__':

    C = Coffee()
    # plot_gray will save a simple image of the original coffee photo and a
    # grayscale version side by side
    C.plot_gray()
    C.plot_hsv()
    # plot_hsv is similar to plot_gray, but this time converting the original
    # rgb image to hsv colormode
    C.threshold_segmentation()
    # A plot with many examples of manual threshold editing
    C.threshold_sk()
    # An example using skimage threshold algorithms

    A = Astronaut()
    # A form of face detection
    A.active_contourning()
    A.chanvese_segmentation()
    A.boundaries()
    A.iterative_cluster_threshold()
    A.felzenszwalb()
    A.felzenszwalb(fname='astro_fezenszwalb_mark.jpg', mark=True)
