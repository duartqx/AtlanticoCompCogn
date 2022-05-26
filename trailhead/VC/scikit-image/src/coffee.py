from os import path
from skimage import data, filters
from skimage.filters import thresholding
from skimage.color import rgb2gray, rgb2hsv
from typing import Callable, TypeAlias, Union
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

#AxImage: TypeAlias = plt.AxesImage
ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'
ImageAny: TypeAlias = Union[ImageBw, ImageColor]

class Coffee:
    def __init__(self, savelocation: str, show: bool=False) -> None:
        self.savelocation = savelocation
        self.coffee: ImageAny = data.coffee()
        self.gcoffee: ImageBw = rgb2gray(self.coffee)
        self.show: bool = show
        self.figure: plt.Figure = plt.figure(figsize=(15, 15))

    @staticmethod
    def _plt_config(*args, **kwargs) -> None:
        ''' Configs subplots and imshow '''
        plt.subplot(*args[0])
        clrbar: AxesImage = plt.imshow(args[1], cmap=kwargs.get('cmap', None))

        if kwargs.get('colorbar', None) is not None:
            plt.colorbar(clrbar, fraction=0.046, pad=0.04)

    def _gray_or_hsv(self, method: Callable, fname: str, **kwargs) -> None:
        _s: ImageAny = method(self.coffee)
        self._plt_config((1, 2, 1), self.coffee)
        self._plt_config((1, 2, 2), _s, **kwargs)
        plt.savefig(path.join(self.savelocation, fname))

    def plot_gray(self, fname: str='plot_gray.jpg') -> None:
        self._gray_or_hsv(rgb2gray, fname, cmap='gray')
        if self.show: plt.show()

    def plot_hsv(self, fname: str='plot_hsv.jpg') -> None:
        self._gray_or_hsv(rgb2hsv, fname)
        if self.show: plt.show()

    def limiar_segmentation(self, fname: str='thresholds.jpg') -> None:
        for i in range(10):
            bin_gray = (self.gcoffee > i*0.1)*1
            plt.subplot(5, 2, i+1)
            plt.title(f'Threshold: > {round(i*0.1,1)}')
            plt.imshow(bin_gray, cmap='gray')
        plt.tight_layout()
        plt.savefig(path.join(self.savelocation, fname))
        if self.show: plt.show()

    @staticmethod
    def _plot_limiar_config(i: int, title: str, img: ImageBw) -> None:
        plt.subplot(2, 2, i)
        plt.title(title)
        plt.imshow(img, cmap='gray')

    def _plot_limiar_sk(self, i: int, 
                        filter: Callable, 
                        title: str,
                        threshold: float=None) -> float:
        if threshold is not None:
            self._plot_limiar_config(i, title, threshold)
        else:
            threshold = filter(self.gcoffee)
            bin_coffee: ImageBw = (self.gcoffee > threshold) * 1
            if filter is thresholding.threshold_otsu:
                # Adds the threshold to threshold_otsu's title
                title += str(threshold)
            self._plot_limiar_config(i, title, bin_coffee)
        return threshold

    def limiar_sk(self, fname: str='limiar_sk.jpg') -> None:
        titles: list[str] = [
                'Threshold: >',
                'Niblack Thresholding', 
                'Sauvola Thresholding', 
                'Sauvola Thresholding - Converting to 0\'s and 1\'s' ]

        sk_filters: list[Callable]; 
        sk_filters = [thresholding.threshold_otsu, 
                      thresholding.threshold_niblack, 
                      thresholding.threshold_sauvola,
                      thresholding.threshold_sauvola]

        threshold = None
        for i, (filter, title) in enumerate(zip(sk_filters, titles), start=1):
            if i == 3:
                threshold = self._plot_limiar_sk(i, filter, title, threshold)
            else:
                threshold = self._plot_limiar_sk(i, filter, title)

        plt.savefig(path.join(self.savelocation, fname))
        if self.show: plt.show()

if __name__ == '__main__':

    C = Coffee('exports')
    C.plot_gray()
    C.limiar_segmentation()
    C.limiar_sk()
