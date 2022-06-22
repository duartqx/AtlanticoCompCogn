from glob import glob
from os import path
from pandas import concat, DataFrame
from .measure import get_df_properties, measure_segmentation_iou, show_props
from .segmentation import Segment


def get_segmentations(origs: list[str],
                      save_dir: str = 'data/exports/all',
                      plot: bool = False) -> None:
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
    print('Initializing Segmentation', flush=True)
    print(f'Found {len(origs)} to segment', flush=True)
    for i, img in enumerate(origs):
        print(f'Segmenting {i}th Image: {img}\n', flush=True)
        segment = Segment(img=img, plot=plot, dir=save_dir)
        segment.isodata()
        segment.canny()
        segment.flood_fill()
        segment.iterative_cluster()
        segment.chanvese()
        segment.sauvola()
        del segment


def get_metrics_df(
        imgs: list[str],
        csv_fname: str = 'metrics.csv') -> DataFrame:
    ''' Builds a pandas dataframe with the metrics axis_major_length,
    axis_minor_length and area of all images on imgs using
    skimage.measure.regionprops_table, saves these metrics to a csv file and
    returns the dataframe '''
    names: dict[str, list[str]] = {
        'name': [path.basename(img) for img in imgs]}
    df: DataFrame = get_df_properties(imgs)
    # Adding names to df
    df = concat([DataFrame(names), df], axis=1)
    df.sort_values(by='name', inplace=True, ignore_index=True)
    df.to_csv(csv_fname)
    return df


def _measure_segmentation(
        imgs_true: list[str],
        tests: list[tuple[str, ...]],
        names: list[str],
        csv_fname: str = 'csvs/compared_precision.csv') -> DataFrame:
    ''' Calculates the precision of the segmentation of tests compared to the
    gold standards imgs_true using skimage.metrics.adapted_rand_error,
    encapsulates these metrics into a pandas dataframe, saves it to a csv file
    and returns it '''
    df: DataFrame = measure_segmentation_iou(imgs_true, tests, names)
    df.to_csv(csv_fname)
    print(df)
    print()
    print(df.sum())
    return df


def get_tests_measures(
        trues: list[str],
        tests_dir: list[str] = ['data/exports/gold/isodata/*.png',
                                'data/exports/gold/canny/*.png',
                                'data/exports/gold/flood_fill/*.png'],
        seg_names: list[str] = ['isodata', 'canny', 'flood_fill']) -> None:
    ''' Measures the best segmentation by calculating their precision, storing
    all the results in a dataframe and summing every column to see which has
    the higher value.
    Important:
        Make sure all images in tests_dir sort in the same order or
        _measure_segmentation will raise an error if the image files don't
        match their shape
        If the images have the same shape, it won't raise an error, but the
        precision will be completelly off
    '''
    assert len(tests_dir) == len(seg_names), \
        'tests_dir and seg_names lengths must match'
    tests_images: list[list[str]]
    tests_images = [sorted(glob(test_dir)) for test_dir in tests_dir]
    # zipping all segmentation of the same original -> [(01, 01),(02, 02)]
    _measure_segmentation(trues, list(zip(*tests_images)), seg_names)


def metrics_n_plot(directory: str = 'data/exports/all/*.png') -> None:
    ''' Quick and dirty function that saves the dataframe with the segmented
    images metrics and plots their images with bounding box and major and minor
    lengths '''
    segs: list[str] = glob(directory)
    get_metrics_df(segs)
    # Loops through segs, loads each image and calculates it's properties
    # axis_major_length, axis_minor_length, area, centroid and boundingbox
    # using skimage.measure.regionprops and plots these informations on the
    # original segmented image using matplotlib.pyplot, it then saves this plot
    # as a new image
    for img in segs:
        show_props(img)


def util(true_dir: str = 'data/gold/*.jpg',
         origs_dir: str = 'data/input/*.jpg',
         **kwargs) -> None:
    '''
    kwargs
        segment (bool) if util must segment the original images
        measure (bool) if util must calculate the segmentation precision
    '''
    trues: list[str] = sorted(glob(true_dir))
    origs: list[str]= sorted(glob(origs_dir))

    if kwargs.get('segment'):
        get_segmentations(origs, plot=True)
    if kwargs.get('measure'):
        get_tests_measures(trues)
    else:
        metrics_n_plot()
