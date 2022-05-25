from src.pimagedit import Simp

def main() -> None:

    ''' This main functions execute tests of some methods in the SIMP
    class. Still incomplete since the class is still a wip '''

    imgedit = Simp(img='resources/ponte.jpg', savelocation='exports')
    imgedit.crop((100, 200))
    imgedit.paint_all((0,0,255))
    imgedit.paint_squares((0,0,255))
    imgedit.flip('h')
    imgedit.rotate(30)
    imgedit.median_blur_montage()
    imgedit.mean_blur_montage()
    imgedit.bilateral_filter_montage()
    imgedit.equalize()
    imgedit.normal_binary_threshold_montage()
    imgedit.adaptive_binary_threshold_montage()
    imgedit.mahotas_montage()
    imgedit.sobel()
    imgedit.laplace()
    imgedit.plt_histogram(color=True, saveplt=True)
    imgedit.savefigs()

    identifying = Simp(img='resources/dados.png', savelocation='exports')
    identifying.identify()

if __name__ == '__main__':

    main()
