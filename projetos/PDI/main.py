from os import path
from src.measure import get_measure, get_df_properties, show_props 
from src.segmentation import Segment

def get_segmentations(input_dir: str='data/input/*.jpg', 
                      save_dir: str='data/exports', 
                      plot: bool=False) -> None:
    ''' Finds all input files and executes four segmentation methods and
    automatically saves then to save_dir '''

    original_imgs = glob(path.join(input_dir, '*.jpg'))

    for img in original_imgs:
        segment = Segment(img=img, plot=plot, dir=save_dir)
        segment.isodata()
        segment.flood_fill()
        segment.iterative_cluster()
        segment.watershed()
        del segment

def plot_props(imgs: list[str]) -> None:
    for img in imgs:
        show_props(img)

def main() -> None:
    pass


if __name__ == '__main__':

    main()

