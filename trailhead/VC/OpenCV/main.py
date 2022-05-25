from src.pimagedit import Simp

def main() -> None:

    ''' This main functions execute tests of some methods in the SIMP
    class. Still incomplete since the class is still a wip '''

    # Testing methods
    imgedit = Simp(img='resources/ponte.jpg', savelocation='exports')
    imgedit.crop((100, 200))
    imgedit.paint_all((0,0,255))
    imgedit.paint_squares((0,0,255))
    imgedit.flip('h')
    imgedit.rotate(30)
    imgedit.median_blur_grid()
    imgedit.mean_blur_grid()
    imgedit.bilateral_filter_grid()
    imgedit.equalize()
    imgedit.normal_binary_threshold_grid()
    imgedit.adaptive_binary_threshold_grid()
    imgedit.mahotas_grid()
    imgedit.sobel()
    imgedit.laplace()
    imgedit.plt_histogram(color=True, saveplt=True)
    imgedit.savefigs()

    # Finding objects
    identifying = Simp(img='resources/dados.png', savelocation='exports')
    identifying.identify()
    identifying.savefigs()

if __name__ == '__main__':

    main()
