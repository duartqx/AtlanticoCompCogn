from src.pimagedit import ImageEditor

def main() -> None:

    ''' This main functions execute tests of some methods in the ImageEditor
    class. Still incomplete since the class is still a wip '''

    imgedit = ImageEditor('resources/ponte.jpg')
    imgedit.crop((100, 200))
    imgedit.paint_all((0,0,255))
    imgedit.paint_squares((0,0,255))
    imgedit.flip('h')
    imgedit.rotate(30)
    imgedit.savefig(location='exports')
    imgedit.plt_histogram(saveplt=True, color=True)

if __name__ == '__main__':

    main()
