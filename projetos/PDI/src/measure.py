import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from os import path
from os.path import basename
from skimage.color import rgb2gray                   # type: ignore
from skimage.measure import label, regionprops, regionprops_table
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.io import imread
from skimage.util import crop
from typing import Any, TypeAlias, Union

ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'
ImageAny: TypeAlias = Union[ImageBw, ImageColor]

def get_image(img: str) -> ImageAny:
    '''
    Loads img with skimage.io.imread and crops it to 880x600 to ensure
    comparissons between gold_standard/img_true have the same shape
    '''
    loaded_img: ImageAny = imread(img, as_gray=True)
    sizes: tuple[int, int, int] = loaded_img.shape
    y: int = abs(880 - sizes[0])//2
    x: int = abs(600 - sizes[1])//2
    # Crop to ensure the images are the same shape
    return crop(loaded_img, ((y, y), (x, x)))

def show_props(
        img: str, dir: str='data/gold/properties', show: bool=False) -> None:
    ''' Function that builds img with it's properties plotted as lines on top
    of the original image, saves it to disk and shows it if show=True
    Source:
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
    '''

    regions = regionprops(label(get_image(img)))
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

def get_regionprops(img: ImageAny) -> pd.DataFrame:
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
        loaded_img: ImageAny = get_image(img)
        img_props: pd.DataFrame = get_regionprops(loaded_img)
        properties = pd.concat([properties, img_props], ignore_index=True)
    return properties

def _get_pr(img_true: ImageAny, 
            img_test: str, 
            r: bool=False) -> tuple[float, float]:
    label_test = label(get_image(img_test))
    _, precision, recall = adapted_rand_error(img_true, label_test)
    if r:
        return precision, recall
    return precision

def get_precisions(imgs_trues: list[str], 
                          imgs_tests: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame()
    for img_true, tests in zip(imgs_trues, imgs_tests):
        label_true = label(get_image(img_true))
        metrics: dict[str, tuple[float, float]] = dict()
        for i, test in enumerate(tests, start=1):
            metric = {f'test_{i}': _get_pr(label_true, test)}
            metrics = metrics | metric # Merges both dictionaries
        df = pd.concat([df, (pd.DataFrame([metrics]))], ignore_index=True)
    return df

if __name__ == '__main__':

    imgs_trues: list[str] = sorted(glob('data/gold/*.jpg'))
    imgs_tests_1: list[str] = sorted(glob('data/gold/segmented/*_isodata.png'))
    #imgs_tests_2: list[str] = sorted(glob('data/gold/segmented/*.png'))
    #imgs_tests_3: list[str] = sorted(glob('data/gold/segmented/*.png'))
    #zip(imgs_tests_1, imgs_tests_2, imgs_tests_3)

    #for img in imgs:
    #    show_props(img)

    #df: pd.DataFrame = get_df_properties(imgs)

    # Adding names to df
    #df = pd.concat([names, isodata_df], axis=1)
    # Sorting df by the name column
    #df = df.sort_values('names', ignore_index=True)

    # Saving to csv
    #df.to_csv('properties.csv')

    df = get_precisions(imgs_trues, [[img] for img in imgs_tests_1])
    print(df)
