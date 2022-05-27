from os import path
from skimage import data, filters
from skimage.filters import thresholding as th
from skimage.color import rgb2gray, rgb2hsv
from typing import Callable, TypeAlias, Union
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'
ImageAny: TypeAlias = Union[ImageBw, ImageColor]

class Coffee:
    def __init__(self, savelocation: str='exports', show: bool=False) -> None:
        self.savelocation = savelocation
        self.coffee: ImageAny = data.coffee()
        self.gray_coffee: ImageBw = rgb2gray(self.coffee)
        self.show: bool = show
        self.fig: plt.Figure = self._set_figure()

    def _savefig(self, fname: str) -> None:
        plt.tight_layout()
        plt.savefig(path.join(self.savelocation, fname))
        if self.show: plt.show()
        # Resets the figure so that after every save they all will look right
        self.fig = self._set_figure()

    @staticmethod
    def _set_figure() -> None:
        return plt.figure(figsize=(15, 15))

    @staticmethod
    def _plt_config(*args, **kwargs) -> None:
        plt.subplot(*args[0])
        clrbar: AxesImage = plt.imshow(args[1], cmap=kwargs.get('cmap', None))
        if kwargs.get('colorbar', None) is not None:
            plt.colorbar(clrbar, fraction=0.046, pad=0.04)

    def _gray_or_hsv(self, method: Callable, fname: str, **kwargs) -> None:
        _s: ImageAny = method(self.coffee)
        self._plt_config((1, 2, 1), self.coffee)
        self._plt_config((1, 2, 2), _s, **kwargs)
        self._savefig(fname)

    def plot_gray(self, fname: str='coffee_plot_gray.jpg') -> None:
        self._gray_or_hsv(rgb2gray, fname, cmap='gray')

    def plot_hsv(self, fname: str='coffee_plot_hsv.jpg') -> None:
        self._gray_or_hsv(rgb2hsv, fname)

    def threshold_segmentation(self, fname: str='coffee_thresholds.jpg') -> None:
        for i in range(10):
            bin_gray = (self.gray_coffee > i*0.1)*1
            plt.subplot(5, 2, i+1)
            plt.title(f'Threshold: > {round(i*0.1,1)}')
            plt.imshow(bin_gray, cmap='gray')
        self._savefig(fname)

    @staticmethod
    def _plot_threshold_config(i: int, title: str, img: ImageBw) -> None:
        plt.subplot(2, 2, i)
        plt.title(title)
        plt.imshow(img, cmap='gray')

    def threshold_sk(self, fname: str='coffee_limiar_sk.jpg') -> None:
        ts: tuple[str, str, tuple[str, str]] 
        ts = ('Threshold: >', 'Niblack Thresholding', 
             ('Sauvola Thresholding', "Sauvola Thresholding - 0's and 1's"))
        skm: tuple[Callable, Callable, Callable];
        skm = (th.threshold_otsu, th.threshold_niblack, th.threshold_sauvola)

        for i, (method, title) in enumerate(zip(skm, ts), start=1):
            threshold = method(self.gray_coffee)
            bin_coffee = (self.gray_coffee > threshold)*1
            if method is th.threshold_sauvola:
                self._plot_threshold_config(i, title[0], threshold)
                self._plot_threshold_config(i+1, title[1], bin_coffee)
            else:
                if title == 'Threshold: >': title += str(threshold)
                self._plot_threshold_config(i, title, bin_coffee)
        self._savefig(fname)
