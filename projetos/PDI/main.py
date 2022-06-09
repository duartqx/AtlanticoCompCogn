from src.util import util

def main(measure: bool=False, segment: bool=False) -> None:
    if measure:
        util(measure=True)
        # measure the gold standard segmentation qualities compared to the
        # manual segmentation
    elif segment:
        util(input_dir='data/all', segment=True, metrics=True)
        # Segments all images on 'data/all' and get's their metrics of
        # axis_major_length, axis_minor_length and area
        # also saves the double version of the images side by side
    else:
        util()
        # grabs all segmented images and plots their metrics 

if __name__ == '__main__':

    main()
