import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
from skimage.measure import label, regionprops, regionprops_table
from skimage.metrics import adapted_rand_error
from skimage.io import imread
from skimage.util import crop
from typing import TypeAlias, Union

ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]' # type: ignore
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]' # type: ignore
ImageAny: TypeAlias = Union[ImageBw, ImageColor]

def _get_image(img: str) -> ImageAny:
    ''' Loads img with skimage.io.imread and crops it to 880x600 to ensure
    comparissons between gold_standard/img_true have the same shape '''
    loaded_img: ImageAny = imread(img, as_gray=True)
    sizes: tuple[int, int, int] = loaded_img.shape
    y: int = abs(880 - sizes[0])//2
    x: int = abs(600 - sizes[1])//2
    # Crop to ensure the images are the same shape
    return crop(loaded_img, ((y, y), (x, x)))

def show_props(
        img: str, dir: str='data/exports/props', show: bool=False) -> None:
    ''' Function that builds img with it's properties plotted as lines on top
    of the original image, saves it to disk and shows it if show=True
    Source:
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
    '''
    loaded_img = label(_get_image(img), connectivity=1)
    regions = regionprops(loaded_img)
    fig, ax = plt.subplots(figsize=(9,16), tight_layout=True)
    ax.imshow(loaded_img, cmap='gray')

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    ax.axis((0, 600, 900, 0))
    plt.savefig(path.join(dir, 'props-'+path.basename(img)))
    if show:
        plt.show()
    plt.close(fig)

def _get_regionprops(img: ImageAny) -> pd.DataFrame:
    ''' Returns a pandas dataframe with the img's metrics for
    axis_major_length, axis_minor_length and it's area '''
    return pd.DataFrame(
            regionprops_table(label(img),
                properties=('axis_major_length', 'axis_minor_length', 'area')))

def get_df_properties(imgs: list[str]) -> pd.DataFrame:
    ''' Loops through all images in imgs, loads then with imread, get it's
    metrics with get_regionprops and returns the concatenated pd.DataFrame with
    the metrics of all images '''
    properties = pd.DataFrame()
    for img in imgs:
        img_props: pd.DataFrame = _get_regionprops(_get_image(img))
        properties = pd.concat([properties, img_props.iloc[0:1]], ignore_index=True)
    return properties

def _get_pr(img_true: ImageAny, 
            img_test: str, 
            r: bool=False) -> Union[tuple[float, float], float]:
    ''' Compares the img_true (gold gold_standard) with img_test using
    skimage.metrics.adapted_rand_error and either returns a tuple of two floats
    with precision and recall or just a float precision if r=False'''
    label_test = label(_get_image(img_test))
    _, precision, recall = adapted_rand_error(img_true, label_test)
    # from adapted_rand_error docstring:
    #- `prec`: float
    #    The adapted Rand precision: this is the number of pairs of pixels that
    #    have the same label in the test label image *and* in the true image,
    #    divided by the number in the test image.
    #- `rec`: float
    #    The adapted Rand recall: this is the number of pairs of pixels that
    #    have the same label in the test label image *and* in the true image,
    #    divided by the number in the true image.
    if r:
        return precision, recall
    return _

def measure_segmentation_precision(imgs_trues: list[str], 
                                   imgs_tests: list[tuple[str, ...]],
                                   test_names: list[str],
                                   r: bool=False) -> pd.DataFrame:
    ''' Builds a pandas dataframe with the precision metric of all imgs_tests
    compared to their img_true and using self._get_pr and returns it '''
    df = pd.DataFrame()
    for img_true, test_imgs in zip(imgs_trues, imgs_tests):
        label_true = label(_get_image(img_true))
        metrics: dict[str, Union[tuple[float, float], float]] = dict()
        for test, name in zip(test_imgs, test_names):
            metric = {name: _get_pr(label_true, test, r=r)}
            metrics = metrics | metric # Merges both dictionaries
        df = pd.concat([df, (pd.DataFrame([metrics]))], ignore_index=True)
        # metrics is inside a list here so that if r=True and metric has tuple
        # as values it keeps those tuples in a single column of a tuple, if
        # metrics is not inside brackets then pd.DataFrame separates the tuple
        # into multiple rows. ignore_index avoids repeated index numbers and
        # resets the index everytime it concatenates the two dataframe
    return df
