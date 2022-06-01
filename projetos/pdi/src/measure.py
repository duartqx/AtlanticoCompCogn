import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from os import path
from os.path import basename
from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imread
from typing import Any, TypeAlias, Union

ImageColor: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
ImageBw: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'
ImageAny: TypeAlias = Union[ImageBw, ImageColor]

def show_props(
        img: str, dir: str='data/gold/properties', show: bool=False) -> None:
    ''' Function that builds img with it's properties plotted as lines on top
    of the original image, saves it to disk and shows it if show=True
    Source:
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
    '''

    loaded_img: ImageBw = imread(img)
    label_img = label(loaded_img)
    regions = regionprops(label_img)
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
        loaded_img: ImageAny = imread(img)
        img_props: pd.DataFrame = get_regionprops(loaded_img)
        properties = pd.concat([properties, img_props], ignore_index=True)
    return properties

if __name__ == '__main__':

    imgs = glob('data/gold/segmented/*.png')

    #for img in imgs:
    #    show_props(img)

    names = pd.DataFrame({'names': [basename(img) for img in imgs]})

    df: pd.DataFrame = get_df_properties(imgs)

    # Adding names to df
    df = pd.concat([names, df], axis=1)
    # Sorting df by the name column
    df = df.sort_values('names', ignore_index=True)

    # Saving to csv
    df.to_csv('properties.csv')
