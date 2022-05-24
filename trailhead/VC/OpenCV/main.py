from src.pimagedit import ImageEditor

def main() -> None:

    ''' This main functions execute tests of some methods in the ImageEditor
    class. Still incomplete since the class is still a wip '''

    imgedit = ImageEditor(img='resources/ponte.jpg', savelocation='exports')
    imgedit.crop((100, 200))
    imgedit.paint_all((0,0,255))
    imgedit.paint_squares((0,0,255))
    imgedit.flip('h')
    imgedit.rotate(30)
    imgedit.median_blur_montage()
    imgedit.mean_blur_montage()
    imgedit.bilateral_filter_montage()
    #imgedit.plt_histogram(equalize=True, saveplt=True)
    imgedit.equalize()
    imgedit.savefigs()

if __name__ == '__main__':

    main()
