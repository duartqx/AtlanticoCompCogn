import numpy as np
import matplotlib.pyplot as plt
import skimage.filters.thresholding as thr
import warnings
from os import path
from scipy import ndimage as ndi
from skimage.color import label2rgb, rgb2gray
from skimage.feature import canny
from skimage.filters.edges import sobel
from skimage.io import imread, imsave
from skimage.morphology import binary_dilation, remove_small_objects
from skimage.segmentation import (chan_vese, felzenszwalb, flood_fill,
                                  mark_boundaries, slic, watershed)
from skimage.util import crop, img_as_ubyte
from typing import Any, Callable, TypeAlias, Union

# type: ignore
ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'  # type: ignore
ImageAny: TypeAlias = Union[ImageBw, ImageColor]


class Segment:
    def __init__(self,
                 img: str, dir: str = 'data/exports',
                 show: bool = False, plot: bool = True,
                 shape: tuple[int, int] = (880, 610)) -> None:
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
        self.shape = shape
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

    def _get_image(self, img: str) -> ImageAny:
        ''' Uses img: str to load the image with skimage.io.imread and also
        crops the image based on self.shape default is (880, 610) '''
        loaded_img: ImageAny = imread(img)
        sizes: tuple[int, int, int] = loaded_img.shape
        if sizes[0] < self.shape[0]:
            return loaded_img
        # if loaded_img is smaller than self.shape, returns it without crop
        y: int = abs(self.shape[0] - sizes[0]) // 2
        x: int = abs(self.shape[1] - sizes[1]) // 2
        return crop(loaded_img, ((y, y), (x, x), (0, 0)))

    def _expname(self, fname: str) -> str:
        return '_'.join((self.expfname, fname))

    def _set_self_fig(self) -> None:
        ''' Sets or resets self.fig so that even after saving the figure
        continues with the same aspect ratio and size '''
        self.fig = plt.figure(figsize=(24, 13.5), tight_layout=True)

    def _savefig(self, fname: str) -> None:
        ''' saves a plotted figure with plt.savefig and resets self.fig '''
        plt.savefig(path.join(self.dir, self._expname(fname)))
        plt.close(self.fig)
        # plt.close avoids warning of too many images opened
        self._set_self_fig()

    def _save(self, t: str, fname: str,
              orig: ImageAny = None, seg: ImageAny = None) -> None:  # type: ignore
        ''' Saves the segmented image after every method. If self.plot is True
        then also saves a side by side version with the original and the
        segmented '''
        if self.plot:
            if orig is None:
                orig = self.gray_img
            # Can't have a self attribute as a default value
            self._plot_double(t=t, fname=fname, orig=orig, seg=seg)
        with warnings.catch_warnings():
            # Avoids annoying warning messages spams when using imsave
            warnings.simplefilter("ignore")
            imsave(path.join(self.dir, self._expname(fname)), seg,
                   check_contrast=False)
        del seg

    def _plot_double(
            self, t: str, fname: str, orig: ImageAny, seg: ImageAny) -> None:
        ''' Plots two images side by side, one being it's original and the
        second it's segmentation and saves this plotted figure to disk as an
        image file '''
        fname = '_'.join(('double', fname))
        axes = self.fig.subplots(1, 2)
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title('Original Image')

        axes[1].imshow(seg, cmap='gray')
        axes[1].set_title(t)

        self._savefig(fname)
        if self.show:
            plt.show()

    def chanvese(self, fname: str = 'chanvese.png') -> ImageBw:
        cvese_astr: 'np.ndarray[np.ndarray[np.bool_]]'  # type: ignore
        cvese_astr = chan_vese(self.gray_img, max_num_iter=100)  # type: ignore
        self._save(t='Chan-vese', fname=fname, seg=cvese_astr)  # type: ignore
        return cvese_astr  # type: ignore

    def boundaries(
            self,
            fname: str = 'boundaries.png',
            n: int = 20) -> ImageBw:
        ''' Mark boundaries segmentation method '''
        segs: Any = slic(self.img, n_segments=n, compactness=1)
        # slic = Simple Linear Iterative Clustering
        bounds = mark_boundaries(self.img, segs)
        self._save(t='Boundaries', fname=fname, orig=self.img, seg=bounds)
        return bounds

    def iterative_cluster(
            self,
            fname: str = 'ict.png',
            n: int = 30) -> ImageAny:
        segs: Any = slic(self.img, n_segments=n, compactness=50)
        clustered = label2rgb(segs, self.img, kind='avg')
        self._save(t='Iterative Cluster Threshold', fname=fname,
                   orig=self.img, seg=clustered)
        return clustered

    def felzenszwalb(self, fname: str = 'felzenszwalb.png') -> ImageAny:
        segs: Any = felzenszwalb(self.img, scale=2, sigma=5, min_size=100)
        marked = mark_boundaries(self.img, segs)
        self._save(t='Felzenszwalb', fname=fname, orig=self.img, seg=marked)
        return marked

    def try_all(self, fname: str = 'try_all.png') -> None:
        ''' Try all segmentation method to subjectively check what segmentation
        algorithm is probably the best one to use '''
        thr.try_all_threshold(self.gray_img, figsize=(13.5, 24), verbose=False)
        self._savefig(fname)

    @staticmethod
    def _remove_holes(img: ImageBw, n: int = 3,
                      rso: bool = True, so_size: int = 128) -> ImageBw:
        ''' This method tries to remove small holes and to unite small lines
        into the bigger segmentation '''
        if rso:
            img = remove_small_objects(img, so_size)
        for _ in range(n):
            img = binary_dilation(img)
        return img

    def _thr(self, title: str, fname: str, method: Callable,
             img: ImageAny = None, save: bool = True, **kwargs) -> ImageBw:  # type: ignore
        ''' Threshold segmentation private method that segmentates using the
        passed skimage thresholding function as the method argument, plus
        self._remove_holes and ndi.binary_fill_holes '''
        if img is None:
            img = self.gaussian
        threshold: float = method(img, **kwargs)
        bin_img: ImageBw = self._remove_holes(img < threshold)
        bin_img = ndi.binary_fill_holes(bin_img)  # type: ignore
        if save:
            self._save(t=title, fname=fname, orig=self.gray_img, seg=bin_img)
        return bin_img

    def otsu(self, fname: str = 'otsu.png') -> ImageBw:
        ''' The otsu thresholding method '''
        return self._thr(title='Otsu', fname=fname, method=thr.threshold_otsu)

    def local(self, fname: str = 'local.png') -> ImageBw:
        ''' Local thresholding method '''
        return self._thr(title='Local', fname=fname,
                         method=thr.threshold_local,
                         block_size=35, offset=2)

    def isodata(self, fname: str = 'isodata.png') -> ImageBw:
        ''' Isodata thresholding method '''
        return self._thr(title='Isodata', fname=fname,
                         method=thr.threshold_isodata)

    def minimum(self, fname: str = 'minimum.png') -> ImageBw:
        ''' Minimum thresholding method '''
        return self._thr(title='Minimum', fname=fname,
                         method=thr.threshold_minimum)

    def watershed(self, fname: str = 'watershed.png') -> ImageAny:
        ''' Watershed segmentation method '''
        seg_img: ImageAny = watershed(sobel(self.gray_img),
                                      markers=468, compactness=1)
        self._save(t='Watershed', fname=fname, seg=seg_img)
        return seg_img

    def sauvola(self, fname: str = 'sauvola.png') -> ImageBw:
        ''' Sauvola edge detection segmentation method '''
        edges = self._thr(img=self.gray_img, title=None, fname=None,  # type: ignore
                          method=thr.threshold_sauvola, save=False)
        filled = ndi.binary_fill_holes(edges)
        self._save(t='Sauvola', fname=fname, seg=filled)  # type: ignore
        return filled  # type: ignore

    def canny(self, fname: str = 'canny.png', fill: bool = True) -> ImageAny:
        ''' Canny edge detection with optional ndi.binary_fill_holes that fills
        these edges with white '''
        edges = canny(self.gaussian, mode='nearest')
        edges = self._remove_holes(edges, rso=False)
        if fill:
            filled = ndi.binary_fill_holes(edges)
            self._save(t='Canny', fname=fname, seg=filled)  # type: ignore
            return filled  # type: ignore
        else:
            self._save(t='Canny', fname=fname, seg=edges)
            return edges

    @staticmethod
    def _find_darkest_pixel(img: ImageBw) -> tuple[int, int]:
        ''' Function that finds the darkest pixel location of an image and
        returns it as a tuple (row, column) '''
        where: tuple['np.ndarray[np.int64]']  # type: ignore
        where = np.where(img == min(img.flatten()))
        return where[0][0], where[1][0]

    def flood_fill(
            self,
            fname: str = 'flood_fill.png',
            tol: int = 100) -> ImageBw:
        ''' flood fill segmentation method. this method first tries to locate
        the darkest pixel on the img to serve as the seed that the flood will
        be applied and them with flood_fill tries to paint the entire leaf in
        white (255). With ndi.binary_fill_holes it tries to close any small
        holes inside only the pixels of value 255 on the flooded img '''
        img = img_as_ubyte(self.gray_img)
        # flood fill needs a seed, or point that it needs to be applied to on
        # an image, so self._find_darkest_pixel tries to locate the darkest
        # pixel on the image, since the darkest one is probably inside a leaf
        seed = self._find_darkest_pixel(img)
        flooded: ImageBw = flood_fill(img, seed, new_value=255, tolerance=tol)
        flooded = ndi.binary_fill_holes(flooded == 255)  # type: ignore
        # flooded == 255 returns a new ndarray where all pixels that were not
        # 255 on the original flooded are now black and the ones that were
        # painted white (255) continue white. making it now a binary image
        flooded = self._remove_holes(flooded, rso=False, n=2)
        self._save(t='Flood_fill', fname=fname, seg=flooded)
        return flooded


if __name__ == '__main__':

    from glob import glob

    imgs = glob('data/input/*.jpg')

    for img in imgs:
        seg = Segment(img, plot=False)
        seg.isodata()
        seg.canny()
        seg.flood_fill()
