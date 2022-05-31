import numpy as np                                              # type: ignore
import matplotlib.pyplot as plt                                 # type: ignore
import skimage.segmentation as skseg                            # type: ignore
import skimage.filters.thresholding as thr                      # type: ignore
import warnings
from glob import glob
from os import path
from skimage.color import label2rgb, rgb2gray                   # type: ignore
from skimage.io import imread, imsave                           # type: ignore
from skimage.util import img_as_ubyte
from typing import Any, Callable, TypeAlias, Union

ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'
ImageAny: TypeAlias = Union[ImageBw, ImageColor]
Snake: TypeAlias = 'np.ndarray[np.ndarray[np.float64]]'

class Segment:
    def __init__(self,
                 img: str, dir: str='data/exports', 
                 show: bool=False, plot: bool=True) -> None:
        ''' 
        Helper class with segmentation methods from scikit-image and autosave
        feature for all methods of images plotted with plt.show
        Args:
            img (str): pathlike string to an image file
            dir (str): pathlike string to a directory
            show (bool): if set to True some methods will show the plotted
            image with matplotlib.pyplot.show
            plot (bool): if set to False saves all images directly without
            plotting side by side with the original with matplotlib
        '''
        self.dir = dir
        self.show = show
        self.plot = plot
        self.expfname = path.splitext(path.basename(img))[0]
        self.__img: ImageAny = imread(img)
        self.__gray_img: ImageBw = rgb2gray(self.img)
        self.gssian: ImageBw = thr.gaussian(self.gray_img, 3)
        self._set_self_fig()

    @property
    def img(self) -> ImageAny:
        return self.__img

    @property
    def gray_img(self) -> ImageBw:
        return self.__gray_img

    def _expname(self, fname: str) -> str:
        return '_'.join((self.expfname, fname))

    def _set_self_fig(self) -> None:
        ''' Sets or resets self.fig so that even after saving the figure
        continues with the same aspect ratio and size '''
        self.fig = plt.figure(figsize=(24, 13.5), tight_layout=True)

    def _savefig(self, fname: str, fig: ImageAny=None) -> None:
        plt.savefig(path.join(self.dir, self._expname(fname)))
        plt.close(self.fig)
        # plt.close avoids warning of too many images opened
        self._set_self_fig()

    def _save(self, t: str, fname: str,
                    orig: ImageAny, seg: ImageAny=None, **kwargs) -> None:
        if self.plot:
            self._plot(t=t, fname=fname, orig=orig, seg=seg)
        else:
            with warnings.catch_warnings():
                # Avoids annoying warning messages spams when using imsave
                warnings.simplefilter("ignore")
                imsave(path.join(self.dir, self._expname(fname)), 
                       seg, check_contrast=False)

    def _plot(self, t: str, fname: str, 
                    orig: ImageAny, seg: ImageAny=None, **kwargs) -> None:
        axes: plt.Axes = self.fig.subplots(1, 2)
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title('Original Image')

        if seg is not None:
            axes[1].imshow(seg, cmap='gray')
            axes[1].set_title(t)
        else:
            # If seg is None then snake and img_snake were passed as keyword
            # arguments to plot active_contourning
            snake = (kwargs['snake'][:, 1], kwargs['snake'][:, 0])
            img_snake = (kwargs['img_snake'][:, 1], kwargs['img_snake'][:, 0])

            axes[1].imshow(self.gray_img, cmap='gray')
            #axes[1].plot(*snake, '--r', lw=5)
            axes[1].plot(*img_snake, '-b', lw=5)
            axes[1].set_title(t)

        self._savefig(fname)
        if self.show: plt.show()

    @staticmethod
    def _get_init_snake() -> Snake:
        ''' skimage.segmentation.active_contour needs a 'snake' argument as the
        initial coordinate for bounding the feature, so first we need to
        build it using this function '''
        s = np.linspace(0, 2 * np.pi, 400)
        r = 100 + 100 * np.sin(s)
        c = 220 + 100 * np.cos(s)
        return np.array([r, c]).T

    def active_contourning(self, fname: str='activecontour.png') -> None:
        snake: Snake = self._get_init_snake()
        img_snake: Snake = skseg.active_contour(self.gray_img, snake)
        #img_snake: Snake = skseg.active_contour(self.gssian, snake)
        self._save(t='Active Contour',fname=fname, orig=self.gray_img, 
                   snake=snake, img_snake=img_snake)

    def chanvese(self, fname: str='chanvese.png') -> ImageBw:
        cvese_astr: tuple['np.ndarray[np.ndarray[np.bool_]]', ...]
        cvese_astr = skseg.chan_vese(self.gray_img, 
                                     max_num_iter=100, 
                                     extended_output=True)
        t: str = f'Chan-vese segmentation - {len(cvese_astr[2])} iterations.'
        self._save(t=t, fname=fname, orig=self.gray_img, seg=cvese_astr[0])
        return cvese_astr[0]

    def boundaries(self, fname: str='boundaries.png', n: int=20) -> ImageBw:
        _segs: Any = skseg.slic(self.img, n_segments=n, compactness=1)
        # slic = Simple Linear Iterative Clustering
        bounds = skseg.mark_boundaries(self.img, _segs)
        self._save(t='Boundaries', fname=fname, orig=self.img, seg=bounds)
        return bounds

    def iterative_cluster(self, fname: str='ict.png', n: int=20) -> ImageAny:
        _segs: Any = skseg.slic(self.img, n_segments=n, compactness=10)
        clustered = label2rgb(_segs, self.img, kind='avg')
        self._save(t='Iterative Cluster Threshold', fname=fname, 
                   orig=self.img, seg=clustered)
        return clustered

    def felzenszwalb(self, 
            fname: str='fezenszwalb.png') -> ImageAny:
        _segs: Any = skseg.felzenszwalb(self.img, 
                                        scale=2, sigma=5, min_size=100)
        marked = skseg.mark_boundaries(self.img, _segs)
        self._save(t='Felzenszwalb', fname=fname, orig=self.img, seg=marked)
        return marked

    def sauvola(self, fname: str='sauvola.png') -> ImageBw:
        threshold: np.ndarray = thr.threshold_sauvola(self.gray_img)
        s_bin: ImageAny = (self.gray_img > threshold) * 1
        self._save(t='Sauvola', fname=fname, orig=self.gray_img, seg=s_bin)
        return s_bin

    def try_all(self, fname: str='try_all.png') -> None:
        thr.try_all_threshold(self.gray_img, figsize=(13.5,24), verbose=False)
        self._savefig(fname)

    def _thr(self, title: str, fname: str,
                   plot: bool, method: Callable, **kwargs) -> ImageBw:
        threshold: float = method(self.gray_img, **kwargs) 
        bin_img: ImageBw = self.gray_img > threshold
        if plot:
            self._save(t=title, fname=fname, orig=self.gray_img, seg=bin_img)
        return bin_img

    def otsu(self, fname: str='otsu.png', **kwargs) -> ImageBw:
        return self._thr(title='Otsu', fname=fname,
                         method=thr.threshold_otsu, 
                         plot=kwargs.get('plot', True))

    def local(self, fname: str='local.png') -> ImageBw:
        return self._thr(title='Local', fname=fname,
                         plot=True, method=thr.threshold_local,
                         block_size=35, offset=2)

    def isodata(self, fname: str='isodata.png') -> ImageBw:
        return self._thr(title='Isodata', fname=fname,
                         plot=True, method=thr.threshold_isodata)

    def minimum(self, fname: str='minimum.png') -> ImageBw:
        return self._thr(title='Minimum', fname=fname,
                         plot=True, method=thr.threshold_minimum)

if __name__ == '__main__':

    # tests

    imgs = glob('data/gold/input/*.jpg')

    for img in imgs:
        segment = Segment(img=img, plot=False)
        ##active_contourning is not working for my leaf photos
        #segment.active_contourning()
        #segment.chanvese()
        #segment.boundaries()
        #segment.iterative_cluster()
        #segment.felzenszwalb()
        #segment.sauvola()
        #segment.try_all()
        #segment.otsu()
        #segment.local()
        segment.isodata()
        #segment.minimum()
