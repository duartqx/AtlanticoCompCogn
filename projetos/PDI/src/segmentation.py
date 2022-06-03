import numpy as np                                              # type: ignore
import matplotlib.pyplot as plt                                 # type: ignore
import skimage.filters.thresholding as thr                      # type: ignore
import warnings
from glob import glob
from os import path
from skimage.color import label2rgb, rgb2gray                   # type: ignore
from skimage.io import imread, imsave                           # type: ignore
from skimage.morphology import binary_dilation, remove_small_objects # type: ignore
from skimage.segmentation import felzenszwalb, slic, chan_vese, mark_boundaries # type: ignore
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

    def _plot(self, t: str, fname: str, orig: ImageAny, seg: ImageAny) -> None:
        axes: plt.Axes = self.fig.subplots(1, 2)
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title('Original Image')

        axes[1].imshow(seg, cmap='gray')
        axes[1].set_title(t)

        self._savefig(fname)
        if self.show: plt.show()

    def chanvese(self, fname: str='chanvese.png') -> ImageBw:
        cvese_astr: tuple['np.ndarray[np.ndarray[np.bool_]]', ...]
        cvese_astr = chan_vese(self.gray_img, 
                                     max_num_iter=100, 
                                     extended_output=True)
        t: str = f'Chan-vese segmentation - {len(cvese_astr[2])} iterations.'
        self._save(t=t, fname=fname, orig=self.gray_img, seg=cvese_astr[0])
        return cvese_astr[0]

    def boundaries(self, fname: str='boundaries.png', n: int=20) -> ImageBw:
        segs: Any = slic(self.img, n_segments=n, compactness=1)
        # slic = Simple Linear Iterative Clustering
        bounds = mark_boundaries(self.img, segs)
        self._save(t='Boundaries', fname=fname, orig=self.img, seg=bounds)
        return bounds

    def iterative_cluster(self, fname: str='ict.png', n: int=50) -> ImageAny:
        segs: Any = slic(self.img, n_segments=n, compactness=10)
        clustered = label2rgb(segs, self.img, kind='avg')
        self._save(t='Iterative Cluster Threshold', fname=fname, 
                   orig=self.img, seg=clustered)
        return clustered

    def felzenszwalb(self, fname: str='fezenszwalb.png') -> ImageAny:
        segs: Any = felzenszwalb(self.img, scale=2, sigma=5, min_size=100)
        marked = mark_boundaries(self.img, segs)
        self._save(t='Felzenszwalb', fname=fname, orig=self.img, seg=marked)
        return marked

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
            img: ImageAny=None, save: bool=True, **kwargs) -> ImageBw:
        if img is None: img=self.gaussian
        threshold: float = method(img, **kwargs) 
        bin_img: ImageBw = self._remove_holes(img < threshold)
        if save:
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

    def _isodata(self) -> ImageBw:
        return self._thr(title='', fname='', save=False,
                         method=thr.threshold_isodata )

    def isodata(self, fname: str='isodata.png') -> ImageBw:
        return self._thr(title='Isodata', fname=fname,
                         method=thr.threshold_isodata)

    def minimum(self, fname: str='minimum.png') -> ImageBw:
        return self._thr(title='Minimum', fname=fname,
                         method=thr.threshold_minimum)

if __name__ == '__main__':

    # tests

    imgs = glob('data/gold/input/01.jpg')

    for img in imgs:
        segment = Segment(img=img, plot=False, dir='data/gold/segmented')
        ##active_contourning is not working for my leaf photos
        segment.active_contourning()
        #local is only giving me black images
        #segment.local()
        #segment.chanvese()
        #segment.boundaries()
        #segment.iterative_cluster()
        #segment.felzenszwalb()
        #segment.sauvola()
        #segment.try_all()
        #segment.otsu()
        #segment.isodata()
        #segment.minimum()
