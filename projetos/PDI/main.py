from glob import glob
from os import path
from pandas import concat, DataFrame
from src.measure import get_df_properties, get_measure, show_props
from src.segmentation import Segment

def get_segmentations(save_dir: str='data/exports', plot: bool=False) -> None:
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

    for img in original_imgs:
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

def measure_precision(imgs_true: list[str], 
                      tests: list[list[str]], 
                      names: list[str]) -> None:
    df = get_measure(imgs_true, tests, names)
    print(df)
    print()
    print(df.sum())

def util(input_dir: str='data/gold', filetype: str='jpg', **kwargs) -> None:
    glob_extension: str = '*.jpg' if filetype == 'jpg' else '*.png'
    # glob_extension is either '*.jpg' or '*.png'

    trues = sorted(glob(path.join(input_dir, glob_extension)))

    if kwargs.get('segment_double', False):
        get_segmentations(trues, plot=True)
    elif kwargs.get('segment', False):
        get_segmentations(trues)
    if kwargs.get('plot_props', False):
        plot_props(imgs)
    if kwargs.get('metrics_df', False):
        get_metrics_df(imgs)
    if kwargs.get('measure_precision', False):

        t_1: list[str] = sorted(glob('data/exports/isodata/*.png'))
        t_2: list[str] = sorted(glob('data/exports/canny/*.png'))
        t_3: list[str] = sorted(glob('data/exports/flood_fill/*.png'))
        t_4: list[str] = sorted(glob('data/exports/ict/*.png'))
        t_5: list[str] = sorted(glob('data/exports/watershed/*.png'))
        t_6: list[str] = sorted(glob('data/exports/felzenszwalb/*.png'))
        tests = list(zip(t_1, t_2, t_3, t_4, t_5, t_6))
        names = ['isodata', 'canny', 'flood_fill', 
                 'ict', 'watershed', 'felzenszwalb']
        measure_precision(trues, list(zip(t_1, t_2, t_3, t_4, t_5, t_6)), names)

        #     isodata     canny  flood_fill  ict  watershed  felzenszwalb
        #0   0.999915  1.000000    1.000000  1.0   0.030148           1.0
        #1   0.999854  0.999912    1.000000  1.0   0.030755           1.0
        #2   0.998050  0.999733    0.998666  1.0   0.022184           1.0
        #3   0.999636  0.999932    0.999954  1.0   0.026042           1.0
        #4   0.999062  1.000000    1.000000  1.0   0.033126           1.0
        #5   0.966275  1.000000    1.000000  1.0   0.102728           1.0
        #6   0.994127  1.000000    1.000000  1.0   0.073994           1.0
        #7   1.000000  1.000000    1.000000  1.0   0.059870           1.0
        #8   0.999183  1.000000    1.000000  1.0   0.103703           1.0
        #9   0.997887  1.000000    1.000000  1.0   0.146720           1.0
        #10  0.992741  1.000000    1.000000  1.0   0.122605           1.0
        #11  0.988155  0.999591    0.988843  1.0   0.040499           1.0
        #12  0.576859  1.000000    1.000000  1.0   0.183090           1.0
        #13  0.994263  1.000000    0.999712  1.0   0.028222           1.0
        #14  0.999193  1.000000    1.000000  1.0   0.124952           1.0
        #15  0.993529  1.000000    0.999066  1.0   0.027426           1.0
        #16  0.999138  1.000000    1.000000  1.0   0.029117           1.0
        #17  0.999838  0.999870    1.000000  1.0   0.033725           1.0
        #18  1.000000  1.000000    1.000000  1.0   0.057231           1.0
        #19  0.999638  1.000000    0.999777  1.0   0.027489           1.0

        #isodata         19.497342
        #canny           19.999038
        #flood_fill      19.986019
        #ict             20.000000
        #watershed        1.303626
        #felzenszwalb    20.000000
        #dtype: float64

def main(measure: bool=False) -> None:
    if measure:
        util(measure_precision=True)
    else:
        util(input_dir='data/all', segment_double=True)


if __name__ == '__main__':

    main()
