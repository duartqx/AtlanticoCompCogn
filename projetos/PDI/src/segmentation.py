import numpy as np
import matplotlib.pyplot as plt
import skimage.filters.thresholding as thr
import warnings
from glob import glob
from os import path
from scipy import ndimage as ndi
from skimage.color import label2rgb, rgb2gray
from skimage.feature import canny
from skimage.filters.edges import sobel
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.morphology import binary_dilation, remove_small_objects
from skimage.segmentation import (chan_vese, felzenszwalb, flood_fill,
                                  mark_boundaries, slic, watershed)
from skimage.util import crop, img_as_ubyte, img_as_uint
from typing import Any, Callable, TypeAlias, Union

ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'
ImageAny: TypeAlias = Union[ImageBw, ImageColor]

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
        if sizes[0] < 880: return loaded_img
        y: int = abs(880 - sizes[0])//2
        x: int = abs(600 - sizes[1])//2
        return crop(loaded_img, ((y, y), (x, x), (0, 0)))

    def _expname(self, fname: str) -> str:
        return '_'.join((self.expfname, fname))

    def _set_self_fig(self) -> None:
        ''' Sets or resets self.fig so that even after saving the figure
        continues with the same aspect ratio and size '''
        self.fig = plt.figure(figsize=(24, 13.5), tight_layout=True)

    def _savefig(self, fname: str) -> None:
        plt.savefig(path.join(self.dir, self._expname(fname)))
        plt.close(self.fig)
        # plt.close avoids warning of too many images opened
        self._set_self_fig()

    def _save(self, t: str, fname: str,
                    orig: ImageAny=None, seg: ImageAny=None, **kwargs) -> None:
        if self.plot:
            if orig is None: orig = self.gray_img
            self._plot(t=t, fname=fname, orig=orig, seg=seg)
        with warnings.catch_warnings():
            # Avoids annoying warning messages spams when using imsave
            warnings.simplefilter("ignore")
            imsave(path.join(self.dir, self._expname(fname)), seg, 
                   check_contrast=False)
        del seg

    def _plot(self, t: str, fname: str, orig: ImageAny, seg: ImageAny) -> None:
        axes: plt.Axes = self.fig.subplots(1, 2)
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title('Original Image')

        axes[1].imshow(seg, cmap='gray')
        axes[1].set_title(t)

        self._savefig(fname)
        if self.show: plt.show()

    def chanvese(self, fname: str='chanvese.png') -> ImageBw:
        cvese_astr: 'np.ndarray[np.ndarray[np.bool_]]'
        cvese_astr = chan_vese(self.gray_img, max_num_iter=100)
        self._save(t='Chan-vese', fname=fname, seg=cvese_astr)
        return cvese_astr

    def boundaries(self, fname: str='boundaries.png', n: int=20) -> ImageBw:
        segs: Any = slic(self.img, n_segments=n, compactness=1)
        # slic = Simple Linear Iterative Clustering
        bounds = mark_boundaries(self.img, segs)
        self._save(t='Boundaries', fname=fname, orig=self.img, seg=bounds)
        return bounds

    def iterative_cluster(self, fname: str='ict.png', n: int=30) -> ImageAny:
        segs: Any = slic(self.img, n_segments=n, compactness=50)
        clustered = label2rgb(segs, self.img, kind='avg')
        self._save(t='Iterative Cluster Threshold', fname=fname, 
                   orig=self.img, seg=clustered)
        return clustered

    def felzenszwalb(self, fname: str='felzenszwalb.png') -> ImageAny:
        segs: Any = felzenszwalb(self.img, scale=2, sigma=5, min_size=100)
        marked = mark_boundaries(self.img, segs)
        self._save(t='Felzenszwalb', fname=fname, orig=self.img, seg=marked)
        return marked

    def try_all(self, fname: str='try_all.png') -> None:
        thr.try_all_threshold(self.gaussian, figsize=(13.5,24), verbose=False)
        self._savefig(fname)

    @staticmethod
    def _remove_holes(img: ImageBw, n: int=3, rso: bool=True) -> ImageBw:
        if rso:
            img = remove_small_objects(img, 128)
        for _ in range(n):
            img = binary_dilation(img)
        return img

    def _thr(self, title: str, fname: str, method: Callable, 
            img: ImageAny=None, save: bool=True, **kwargs) -> ImageBw:
        if img is None: img=self.gaussian
        threshold: float = method(img, **kwargs) 
        bin_img: ImageBw = self._remove_holes(img < threshold)
        bin_img = ndi.binary_fill_holes(bin_img)
        if save:
            self._save(t=title, fname=fname, orig=self.gray_img, seg=bin_img)
        return bin_img

    def otsu(self, fname: str='otsu.png') -> ImageBw:
        return self._thr(title='Otsu', fname=fname, method=thr.threshold_otsu)

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

    def watershed(self, fname: str='watershed.png') -> ImageAny:
        seg_img: ImageAny = watershed(sobel(self.gray_img), 
                                      markers=468, compactness=0.001)
        self._save(t='Watershed', fname=fname, seg=seg_img)
        return seg_img

    def sauvola(self, fname: str='sauvola.png') -> ImageBw:
        edges = self._thr(img=self.gray_img, title=None, fname=None,
                          method=thr.threshold_sauvola, save=False)
        filled = ndi.binary_fill_holes(edges)
        self._save(t='Sauvola', fname=fname, seg=filled)
        return filled

    def canny(self, fname: str='canny.png') -> ImageAny:
        edges = self._remove_holes(canny(self.gaussian, mode='nearest'), 
                                   rso=False)
        filled = ndi.binary_fill_holes(edges)
        self._save(t='Canny', fname=fname, seg=filled)
        return filled

    @staticmethod
    def _find_darkest_pixel(img: ImageBw) -> tuple[int, int]:
        ''' Function that finds the darkest pixel location and returns it as a
        tuple (row, column) '''
        where: tuple['np.ndarray[np.int64]'] 
        where = np.where(img == min(img.flatten()))
        return where[0][0], where[1][0]

    def flood_fill(self, fname: str='flood_fill.png', tol: str=100) -> ImageBw:
        img = img_as_ubyte(self.gray_img)
        # Finds the location of the darkest pixel 
        seed = self._find_darkest_pixel(img)
        flooded: ImageBw = flood_fill(img, seed, new_value=255, tolerance=tol)
        flooded = self._remove_holes(ndi.binary_fill_holes(flooded == 255), 
                                     rso=False, n=2)
        self._save(t='Flood_fill', fname=fname, seg=flooded)
        return flooded
        

if __name__ == '__main__':

    # tests

    imgs = glob('data/input/*.jpg')

    for img in imgs:
        segment = Segment(img=img, plot=False, dir='data/exports')
        ##active_contourning is not working for my leaf photos
        #local is only giving me black images
        #segment.local()
        #segment.chanvese()
        #segment.boundaries()
        #segment.iterative_cluster(n=1000)
        segment.felzenszwalb()
        #segment.sauvola()
        #segment.try_all()
        #segment.otsu()
        #segment.isodata()
        #segment.minimum()
        #segment.watershed()
        #segment.canny()
        #segment.flood_fill()
        del segment
