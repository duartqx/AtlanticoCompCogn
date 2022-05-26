from src.coffee import Coffee

if __name__ == '__main__':

    C = Coffee(savelocation='exports')
    C.plot_gray()
    C.plot_hsv()
    C.threshold_segmentation()
    C.threshold_sk()
