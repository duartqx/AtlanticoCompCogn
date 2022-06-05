from src.util import util

def main(measure: bool=False) -> None:
    if measure:
        util(measure=True)
    else:
        util(input_dir='data/all', segment=True, metrics=True)
        # Segments all images on 'data/all' and get's their metrics of
        # axis_major_length, axis_minor_length and area

if __name__ == '__main__':

    main()
