import numpy as np                                              # type: ignore
import matplotlib.pyplot as plt                                 # type: ignore
import skimage.segmentation as skseg                            # type: ignore
from os import path
from skimage.color import label2rgb, rgb2gray                   # type: ignore
from skimage import data, img_as_float                          # type: ignore
from skimage.filters.thresholding import gaussian               # type: ignore
from typing import Any, Callable, TypeAlias, Union

ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'
ImageAny: TypeAlias = Union[ImageBw, ImageColor]
Snake: TypeAlias = 'np.ndarray[np.ndarray[np.float64]]'

class Astronaut:
    def __init__(self, savelocation: str='exports', show: bool=False) -> None:
        self.savelocation = savelocation
        self.show = show
        self.astronaut: ImageAny = data.astronaut()
        self.gray_astr: ImageBw = rgb2gray(self.astronaut)
        self.gssian: ImageBw = gaussian(self.gray_astr, 3)
        self.fig: plt.Figure = self._set_figure()

    def savefig(self, fname: str) -> None:
        plt.savefig(path.join(self.savelocation, fname))
        if self.show: plt.show()
        self.fig = self._set_figure()

    @staticmethod
    def _set_figure() -> None:
        return plt.figure(figsize=(15, 15))

    @staticmethod
    def _get_init_snake() -> Snake:
        ''' skimage.segmentation.active_contour needs a 'snake' argument as the
        initial coordinate for bounding the feature, so first we need to
        build it using this function '''
        s = np.linspace(0, 2 * np.pi, 400)
        r = 100 + 100 * np.sin(s)
        c = 220 + 100 * np.cos(s)
        return np.array([r, c]).T

    def active_contourning(self, fname: str='astro_activecontour.jpg') -> None:
        snake: Snake = self._get_init_snake()
        astronaut_snake: Snake = skseg.active_contour(self.gssian, snake)
        ax = self.fig.add_subplot(111)
        ax.imshow(self.gray_astr, cmap='gray')
        ax.plot(snake[:, 1], snake[:, 0], '--r', lw=5)
        ax.plot(astronaut_snake[:, 1], astronaut_snake[:, 0], '-b', lw=5)
        self.savefig(fname)

    def chanvese_segmentation(self, fname: str='astro_chanvese.jpg') -> None:
        cvese_astr: tuple['np.ndarray[np.ndarray[np.bool_]]', ...]
        cvese_astr = skseg.chan_vese(self.gray_astr, 
                                     max_num_iter=100, 
                                     extended_output=True)
        axes: list[plt.Axes] = self.fig.subplots(1, 3)

        axes[0].imshow(self.gray_astr, cmap='gray')
        axes[0].set_title('Original Image')
        
        axes[1].imshow(cvese_astr[0], cmap='gray')
        t: str = f'Chan-vese segmentation - {len(cvese_astr[2])} iterations.'
        axes[1].set_title(t)
        
        axes[2].imshow(cvese_astr[1], cmap='gray')
        axes[2].set_title('Final Level Set')

        self.savefig(fname)

    def boundaries(self, fname: str='astro_markboundaries.jpg') -> None:
        _segs: Any = skseg.slic(self.astronaut, n_segments=100, compactness=1)
        # slic = Simple Linear Iterative Clustering
        # Plots two imgs side by side with subplots
        axes: list[plt.Axes] = self.fig.subplots(1, 2)
        axes[0].imshow(self.astronaut)
        axes[1].imshow(skseg.mark_boundaries(self.astronaut, _segs))
        self.savefig(fname)

    def iterative_cluster_threshold(self, fname: str='astro_ict.jpg') -> None:
        _segs: Any = skseg.slic(self.astronaut, n_segments=50, compactness=10)
        axes: list[plt.Axes] = self.fig.subplots(1, 2)
        axes[0].imshow(self.astronaut)
        axes[1].imshow(label2rgb(_segs, self.astronaut, kind='avg'))
        self.savefig(fname)

    def felzenszwalb(self, 
            fname: str='astro_fezenszwalb.jpg', mark: bool=False) -> None:
        _segs: Any = skseg.felzenszwalb(self.astronaut, 
                                        scale=2, sigma=5, min_size=100)
        if mark:
            _segs = skseg.mark_boundaries(self.astronaut, _segs)
        axes: list[plt.Axes] = self.fig.subplots(1, 2)
        axes[0].imshow(self.astronaut)
        axes[1].imshow(_segs, cmap='gray')
        self.savefig(fname)
