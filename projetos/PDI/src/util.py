from glob import glob
from os import path
from pandas import concat, DataFrame
from measure import get_df_properties, get_measure, show_props
from segmentation import Segment

def get_segmentations(origs: list[str], 
                      save_dir: str='data/exports/all', 
                      plot: bool=False) -> None:
    ''' Finds all input files and executes four segmentation methods and
    automatically saves then to save_dir 
    Args
        input_dir (str) pathlike, the directory that contains the images to be
        segmented
        filetype (str) jpg or png, are the filetype of the pictures on the
        input_dir.
        save_dir (str) pathlike, the direcotry the segmented images will be
        saved
        plot (bool) if True the images are plotted side by side with their
        original instead of being saved as a single image
    '''

    for img in origs:
        segment = Segment(img=img, plot=plot, dir=save_dir)
        #segment.isodata()
        segment.canny()
        #segment.flood_fill()
        #segment.iterative_cluster()
        #segment.watershed()
        #segment.felzenszwalb()
        del segment

def plot_props(imgs: list[str]) -> None:
    for img in imgs:
        show_props(img)

def get_metrics_df(imgs: list[str], csv_fname: str='metrics.csv') -> None:

    names = {'name': [path.basename(img) for img in imgs]}
    df: DataFrame = get_df_properties(imgs)
    # Adding names to df
    df = concat([DataFrame(names), df], axis=1)
    df.to_csv(csv_fname)

def _measure(imgs_true: list[str], 
             tests: list[list[str]], 
             names: list[str]) -> None:
    df = get_measure(imgs_true, tests, names)
    df.to_csv('csvs/compared_precision.csv')
    print(df)
    print()
    print(df.sum())

def get_tests_measures(trues: list[str]) -> None:
    ''' Measures the best segmentation by calculating their precision, storing
    all the results in a dataframe and summing every column to see which has
    the higher value, representing the best segmentation '''
    # The six methods that _measure will test
    t_1: list[str] = sorted(glob('data/exports/isodata/*.png'))
    t_2: list[str] = sorted(glob('data/exports/canny/*.png'))
    t_3: list[str] = sorted(glob('data/exports/flood_fill/*.png'))
    t_4: list[str] = sorted(glob('data/exports/ict/*.png'))
    t_5: list[str] = sorted(glob('data/exports/watershed/*.png'))
    t_6: list[str] = sorted(glob('data/exports/felzenszwalb/*.png'))
    # zipping all segmentation of the same original -> [(01, 01,(02, 02)]
    tests = list(zip(t_1, t_2, t_3, t_4, t_5, t_6))
    names = ['isodata', 'canny', 'flood_fill', 
             'ict', 'watershed', 'felzenszwalb']
    _measure(trues, tests, names)

    #isodata         19.497342
    #canny           19.999038 <- The one to use
    #flood_fill      19.986019
    #ict             20.000000 <-
    #watershed        1.303626
    #felzenszwalb    20.000000 <-
    #dtype: float64
    # ict and felzenszwalb are so high it kinds of makes me uncertain that
    # precision is working correctly, so we'll use canny method instead

def metrics_n_plot(directory: str='data/exports/all/*_canny.png') -> None:
    ''' Quick and dirty function that saves the dataframe with canny segmented
    images metrics and plots their images with bounding box and major and minor
    lengths '''
    cannys: list[str] = glob(directory)
    get_metrics_df(cannys)
    plot_props(cannys)

def util(**kwargs) -> None:
    '''
    kwargs
        segment (bool) if util must segment the original images
        measure (bool) if util must calculate the segmentation precision
        metrics (bool) if util must plot the metrics and save them in a df
    '''

    origs: list[str] ; trues: list[str]
    trues = sorted(glob('data/gold/*.jpg'))
    origs = sorted(glob('data/all/*.jpg'))

    if kwargs.get('segment'):
        get_segmentations(origs, plot=True)

    if kwargs.get('measure'):
        get_tests_measures(trues)
    elif kwargs.get('metrics'):
        metrics_n_plot()
