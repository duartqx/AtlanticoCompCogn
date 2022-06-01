import numpy as np                                              # type: ignore
import matplotlib.pyplot as plt                                 # type: ignore
import skimage.segmentation as skseg                            # type: ignore
import skimage.filters.thresholding as thr                      # type: ignore
import warnings
from glob import glob
from os import path
from skimage.color import label2rgb, rgb2gray                   # type: ignore
from skimage.io import imread, imsave                           # type: ignore
from skimage.morphology import (binary_dilation, 
                                remove_small_objects) # type: ignore
from skimage.util import crop, img_as_ubyte
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
        self.__img: ImageAny = self._get_image(img)
        self.__gray_img: ImageBw = rgb2gray(self.img)
        self.gaussian: ImageBw = thr.gaussian(self.gray_img, 3)
        self._set_self_fig()

    @property
    def img(self) -> ImageAny:
        return self.__img

    @property
    def gray_img(self) -> ImageBw:
        return self.__gray_img

    @staticmethod
    def _get_image(img: str) -> ImageAny:
        loaded_img: ImageAny = imread(img)
        sizes: tuple[int, int, int] = loaded_img.shape
        y: int = abs(880 - sizes[0])//2
        x: int = abs(600 - sizes[1])//2
        return crop(loaded_img, ((y, y), (x, x), (0, 0)))

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
        img_snake: Snake = skseg.active_contour(self.gaussian, snake)
        self._plot(t='Active Contour',fname=fname, orig=self.gray_img, 
                   snake=snake, img_snake=img_snake)

    def chanvese(self, fname: str='chanvese.png') -> ImageBw:
        cvese_astr: tuple['np.ndarray[np.ndarray[np.bool_]]', ...]
        cvese_astr = skseg.chan_vese(self.gray_img, 
                                     max_num_iter=100, 
                                     extended_output=True)
        t: str = f'Chan-vese segmentation - {len(cvese_astr[2])} iterations.'
        self._save(t=t, fname=fname, orig=self.gray_img, seg=cvese_astr[0])
        return cvese_astr[0]

    def _segment(self, t: str, fname: str, n: int, 
            method: Callable, c: int=1, **kwargs) -> ImageAny:
        segs: Any; segmented: ImageAny
        if kwargs.get('felzen', None) is not None:
            segs = skseg.felzenszwalb(self.img, scale=2, sigma=5, min_size=100)
        else:
            segs = skseg.slic(self.img, n_segments=n, compactness=c)
        if kwargs.get('ict', None) is not None:
            segmented = label2rgb(segs, self.img, kind='avg')
        else:
            segmented = img_as_ubyte(method(self.img, segs))
        self._save(t=t, fname=fname, orig=self.img, seg=segmented)
        return segmented

    def boundaries(self, fname: str='boundaries.png', n: int=20) -> ImageBw:
        return self._segment(t='Boundaries', fname=fname, 
                             n=n, method=skseg.mark_boundaries)

    def iterative_cluster(self, fname: str='ict.png', n: int=50) -> ImageAny:
        return self._segment(t='Iterative Cluster Threshold', fname=fname,
                             n=n, c=20, method=None, ict=True)

    def felzenszwalb(self, fname: str='fezenszwalb.png') -> ImageAny:
        return self._segment(t='Felzenszwalb', fname=fname, n=None, 
                             method=skseg.felzenszwalb, felzen=True)

    def try_all(self, fname: str='try_all.png') -> None:
        thr.try_all_threshold(self.gaussian, figsize=(13.5,24), verbose=False)
        self._savefig(fname)

    @staticmethod
    def _remove_holes(img: ImageBw) -> ImageBw:
        img = remove_small_objects(img, 128)
        for _ in range(3):
            img = binary_dilation(img)
        return img

    def _thr(self, title: str, fname: str, method: Callable, 
             img: ImageAny=None, **kwargs) -> ImageBw:
        if img is None: img=self.gaussian
        threshold: float = method(img, **kwargs) 
        bin_img: ImageBw = self._remove_holes(img < threshold)
        self._save(t=title, fname=fname, orig=self.gray_img, seg=bin_img)
        return bin_img

    def sauvola(self, fname: str='sauvola.png') -> ImageBw:
        return self._thr(img=self.gray_img, title='Sauvola', fname=fname,
                         method=thr.threshold_sauvola)

    def otsu(self, fname: str='otsu.png') -> ImageBw:
        return self._thr(title='Otsu', fname=fname,
                         method=thr.threshold_otsu, )

    def local(self, fname: str='local.png') -> ImageBw:
        return self._thr(title='Local', fname=fname,
                         method=thr.threshold_local,
                         block_size=35, offset=2)

    def isodata(self, fname: str='isodata.png') -> ImageBw:
        return self._thr(title='Isodata', fname=fname,
                         method=thr.threshold_isodata)

    def minimum(self, fname: str='minimum.png') -> ImageBw:
        return self._thr(title='Minimum', fname=fname,
                         method=thr.threshold_minimum)

if __name__ == '__main__':

    # tests

    imgs = glob('data/gold/input/*.jpg')

    for img in imgs:
        segment = Segment(img=img, plot=False, dir='data/gold/segmented')
        ##active_contourning is not working for my leaf photos
        #segment.active_contourning()
        #local is only giving me black images
        #segment.local()
        #segment.chanvese()
        #segment.boundaries()
        #segment.iterative_cluster()
        #segment.felzenszwalb()
        #segment.sauvola()
        #segment.try_all()
        #segment.otsu()
        segment.isodata()
        #segment.minimum()
