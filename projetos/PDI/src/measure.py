from __future__ import annotations
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imread
from skimage.util import crop
from typing import Union


def _get_image(img: str) -> 'np.ndarray':
    ''' Loads img with skimage.io.imread and crops it to 880x600 to ensure
    comparissons between gold_standard/img_true have the same shape '''
    loaded_img: 'np.ndarray' = imread(img, as_gray=True)
    sizes: tuple[int, int, int] = loaded_img.shape
    y: int = abs(880 - sizes[0]) // 2
    x: int = abs(600 - sizes[1]) // 2
    # Crop to ensure the images are the same shape
    return crop(loaded_img, ((y, y), (x, x)))


def show_props(
        img: str, dir: str = 'data/exports/props', show: bool = False) -> None:
    ''' Function that builds img with it's properties plotted as lines on top
    of the original image, saves it to disk and shows it if show=True
    Source:
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
    '''
    loaded_img = label(_get_image(img), connectivity=1)
    regions = regionprops(loaded_img)
    fig, ax = plt.subplots(figsize=(9, 16), tight_layout=True)
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
    plt.savefig(path.join(dir, 'props-' + path.basename(img)))
    if show:
        plt.show()
    plt.close(fig)


def _get_regionprops(img: 'np.ndarray') -> pd.DataFrame:
    ''' Returns a pandas dataframe with the img's metrics for
    axis_major_length, axis_minor_length and it's area '''
    return pd.DataFrame(
        regionprops_table(
            label(img),
            properties=(
                'axis_major_length',
                'axis_minor_length',
                'area')))


def get_df_properties(imgs: list[str]) -> pd.DataFrame:
    ''' Loops through all images in imgs, loads then with imread, get it's
    metrics with get_regionprops and returns the concatenated pd.DataFrame with
    the metrics of all images '''
    properties = pd.DataFrame()
    for img in imgs:
        img_props: pd.DataFrame = _get_regionprops(_get_image(img))
        properties = pd.concat(
            [properties, img_props.iloc[0:1]], ignore_index=True)
    return properties


def _get_iou(img_true: 'np.ndarray', img_test: 'np.ndarray') -> float:
    '''Metrica IOU: Se a previsão estiver completamente correta, IoU = 1. 
    Quanto menor a IoU, pior será o resultado da previsão.'''
    inter = np.logical_and(img_true, img_test)
    union = np.logical_or(img_true, img_test)
    iou_score = np.sum(inter) / np.sum(union)
    return iou_score


def measure_segmentation_iou(imgs_trues: list[str],
                             imgs_tests: list[tuple[str, ...]],
                             test_names: list[str]) -> pd.DataFrame:
    ''' Builds a pandas dataframe with the precision metric of all imgs_tests
    compared to their img_true and using self._get_pr and returns it '''
    df = pd.DataFrame()
    for img_true, test_imgs in zip(imgs_trues, imgs_tests):
        loaded_true = _get_image(img_true)
        all_metrics = dict()
        for test, name in zip(test_imgs, test_names):
            metric = {name: _get_iou(loaded_true, _get_image(test))}
            all_metrics = all_metrics | metric  # Merges both dictionaries
        df = pd.concat([df, (pd.DataFrame([all_metrics]))], ignore_index=True)
        # metrics is inside a list here so that if r=True and metric has tuple
        # as values it keeps those tuples in a single column of a tuple, if
        # metrics is not inside brackets then pd.DataFrame separates the tuple
        # into multiple rows. ignore_index avoids repeated index numbers and
        # resets the index everytime it concatenates the two dataframe
    return df


if __name__ == '__main__':

    from glob import glob

    golds = sorted(glob('data/gold/*.jpg'))
    names_df = pd.DataFrame({'names': [path.basename(img) for img in golds]})
    df = pd.concat([names_df, get_df_properties(golds)], axis=1)
    df.to_csv('gold-major-minor-length-area-with_names.csv')
