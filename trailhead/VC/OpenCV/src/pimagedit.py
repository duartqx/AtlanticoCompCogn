from numpy import ndarray, uint8, zeros
from queue import Queue
from typing import Optional, TypeAlias, Union
import cv2
import matplotlib.pyplot as plt

BGRTuple: TypeAlias = tuple[uint8, uint8, uint8]
CVImage: TypeAlias = 'ndarray[ndarray[ndarray[uint8]]]'
RotationMatrix: TypeAlias = 'ndarray[ndarray[float64]]'

class ImageEditor:
    ''' Python extremely simple image editor using OpenCV '''
    def __init__(self, filename: str, verb: bool=False) -> None:
        self.img_name_n_ext: list[str] = filename.split('/')[-1].split('.')
        # Assert that img_name_n_ext is a list of two strings
        assert len(self.img_name_n_ext) == 2, 'Invalid filename'
        self.img_name: str = self.img_name_n_ext[0]
        self.img_ext: str = self.img_name_n_ext[1]
        # Reads the image
        self.img: CVImage = cv2.imread(filename)
        # Initializes last_edited variable to None and starts a Queue 
        self.last_edited: CVImage = None
        self.edited_imgs: Queue[CVImage] = Queue()
        self.verb = verb

    def __len__(self) -> int:
        return self.edited_imgs.qsize()

    def __repr__(self) -> str:
        return f'''
        Editing {self.img}
        You have {self.edited_imgs.qsize()} unsaved edited images'''

    def _put_last(self, *args, operation: str='') -> None:
        ''' Adds edted_img to self.edited_imgs Queue and sets
        self.last_edited to edted_img '''
        for edted_img in args:
            self.edited_imgs.put(edted_img)
            self.last_edited = edted_img
        if self.verb:
            print(f'{operation} image file and added it to the save Queue')

    @staticmethod
    def _savefig(flname: str, fig: CVImage) -> None:
        cv2.imwrite(flname, fig)

    def savefig(self, location: str=None) -> None:
        ''' Saves the edited image to an image file '''
        if location is not None:
            assert isinstance(location, str), 'Directory must be a string'
            if not location.endswith('/'):
                location += '/'
        else:
            location = ''

        for i in range(self.edited_imgs.qsize()):
            flname: str = f'{location}{self.img_name}_ed_{i+1}.{self.img_ext}'
            self._savefig(flname, self.edited_imgs.get())
        if self.verb:
            print('Saved all edited images')

    def show(self) -> None:
        ''' Shows last edited image '''
        cv2.imshow('Edited Image', self.last_edited)
        cv2.waitKey(0)

    def _paint(self, color: BGRTuple, brush: str, step: int=1) -> CVImage:
        ''' Paint pixels of an image with a flat color or squares of a single
        color'''
        edted_img: CVImage = self.img.copy()
        for y in range(0, edted_img.shape[0], step):
            for x in range(0, edted_img.shape[1], step):
                if brush == 'all':
                    edted_img[y, x] = color
                elif brush == 'squares':
                    edted_img[y:y+5, x:x+5] = color
        self._put_last(edted_img, operation='Painted')

    def paint_all(self, color: BGRTuple) -> CVImage:
        ''' Completelly paints the image with one rgb color (bgr_color) '''
        return self._paint(color=color, brush='all')

    def paint_squares(self, color: BGRTuple) -> CVImage:
        ''' Paints the image with squares of one color '''
        return self._paint(color=color, brush='squares', step=10)

    def crop(self, size: tuple[int, int]) -> CVImage:
        ''' 
        Crops the image to a new (size) 
        Example: crop_image(img, size=(100, 200)) -> img[100:200, 100:200]
        Args:
            size (tuple[int, int]) tuple with the new size for the image
        '''
        cropped_img: CVImage = self.img[size[0]:size[1], size[0]:size[1]]
        self._put_last(cropped_img, operation='Cropped')
        return cropped_img

    def resize(self, new_width: int) -> CVImage:
        ''' 
        Resizes the image to the new_width in proportion to the new_height
        '''
        old_h: int; old_w: int
        old_h, old_w = self.img.shape[:2]
        new_height: int = new_width * (old_h//old_w)
        new_size: tuple[int, int] = (new_width, new_height)
        resized_img: CVImage = cv2.resize(self.img.copy(), 
                                          new_size, 
                                          interpolation=cv2.INTER_AREA)
        self._put_last(resized_img, operation='Resized')
        return resized_img

    def flip(self, axis: Union[str, int]) -> CVImage:
        ''' Flips the image horizontally, vertically or both 
        Arg:
            axis (Union[str, int]) needs to be set as:
                'horizontal', 'h', 'hor', '1' or 1 to flip horizontally
                'vertical', 'v', 'ver', '0'or 0 to flip vertically
                or 'a', 'all', 'b', 'both', '-1' or -1 to flip both 
        '''
        ax: int
        if axis in ('horizontal', 'h', 'hor', '1', 1): ax = 1
        elif axis in ('vertical', 'v', 'ver', '0', 0): ax = 0
        elif axis in ('a', 'all', 'b', 'both', '-1', -1): ax = -1
        else: 
            raise ValueError('''
            You need to pass a valid string or int as axis.
            'horizontal', 'h', 'hor', '1' or 1 to flip horizontally
            'vertical', 'v', 'ver', '0'or 0 to flip vertically
            'a', 'all', 'b', 'both', '-1' or -1 to flip both 
            vertically and horizontally
            ''')
        flipped_img: CVImage = cv2.flip(self.img.copy(), ax)
        self._put_last(flipped_img, operation='Flipped')
        return flipped_img

    def rotate(self, d: int) -> CVImage:
        ''' Rotates the image related to the center by (d) degrees '''
        h: int; w: int
        h, w = self.img.shape[:2] # Grabs width and height of self.img
        center: tuple[int, int] = (w // 2, h // 2) # Finds the center
        rot_matrix: RotationMatrix = cv2.getRotationMatrix2D(center, d, 1.0)
        rotated_img: CVImage = cv2.warpAffine(self.img, rot_matrix, (w, h))
        self._put_last(rotated_img, operation='Rotated')
        return rotated_img

    def colormode(self, color_format: str) -> CVImage:
        ''' Alters self.img from bgr color format to gayscale, hsv or lab 
        Arg:
            color_format (str) Can be set to:
                'gray' to convert to a grayscale version of self.img
                'hsv' to convert to HSV version
                'lab' to converto to lab color format
        '''
        altered_img: CVImage = self.img.copy()
        if color_format == 'gray':
            altered_img = cv2.cvtColor(altered_img, cv2.COLOR_BGR2GRAY)
        elif color_format == 'hsv':
            altered_img = cv2.cvtColor(altered_img, cv2.COLOR_BGR2HSV)
        elif color_format == 'lab':
            altered_img = cv2.cvtColor(altered_img, cv2.COLOR_BGR2LAB)
        self._put_last(altered_img, operation='Altered color mode from')
        return altered_img

    def split(self, *args) -> tuple[CVImage, CVImage, CVImage]:
        ''' Splits self.img color channels and adds them to the save Queue'''
        blue_ch: CVImage; green_ch: CVImage; red_ch: CVImage;
        blue_ch, green_ch, red_ch = cv2.split(self.img)
        if not args:
            self._put_last(blue_ch, green_ch, red_ch, operation='Splitted')
        return blue_ch, green_ch, red_ch

    @staticmethod
    def merge(b_ch: CVImage, 
              g_ch: CVImage, 
              r_ch: CVImage, 
              savefig=False) -> CVImage:
        ''' Merges three separated color channels into a single image file '''
        merged_chs: CVImage = cv2.merge([b_ch, g_ch, r_ch])
        if savefig:
            ImageEditor._savefig('merged_img.png', merged_chs)
        return merged_chs

    @staticmethod
    def merge_single_channel(color_ch: CVImage, 
                             color: str, 
                             savefig: bool=False) -> CVImage:
        zeros_ch: CVImage = zeros(color_ch.shape[:2], dtype='uint8')
        merged_img: CVImage
        if color == 'red':
            merged_img = cv2.merge([zeros_ch, zeros_ch, color_ch])
        elif color == 'green':
            merged_img = cv2.merge([zeros_ch, color_ch, zeros_ch])
        elif color == 'blue':
            merged_img = cv2.merge([color_ch, zeros_ch, zeros_ch])
        if savefig:
            ImageEditor._savefig('merged_single_channel.png', merged_img)
        return merged_img

    @staticmethod
    def plt_config(colormode: str, color_channels: tuple[CVImage]) -> None:
        plt.figure()
        plt.title(f'Histogram {colormode}')
        plt.xlabel('Intensity')
        plt.ylabel('Qtty of Pixels')
        colors: list[str]
        if len(color_channels) == 1: colors = ['gray']
        else: colors = ['b','g','r']
        for c_ch, color in zip(color_channels, colors):
            hist: 'ndarray[float32]'= ImageEditor.calc_hist(c_ch)
            plt.plot(hist, color=color)
            plt.xlim([0, 256])


    @staticmethod
    def calc_hist(img: CVImage) -> 'ndarray[float32]':
        ''' Returns the np array with the img histogram '''
        return cv2.calcHist([img], [0], None, [256], [0, 256])

    def plt_histogram(self, **kwargs) -> None:
        '''
        Plots self.img histogram
        **kwargs:
            ravel: bool = False
            color: bool = False
            saveplt: bool = False
        '''
        if 'color' not in kwargs:
            # Plot the single B&W histogram
            g_img: tuple[CVImage]
            g_img = (cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2GRAY),)
            if 'ravel' in kwargs:
                # Plots ravel histogram
                plt.hist(g_img[0].ravel(), 256, [0, 256])
            else:
                # Plots line histogram for b&w image
                self.plt_config('B&W', g_img) 
        else:
            # Plots three histogram for each color channel
            c_chs: tuple[CVImage, CVImage, CVImage] = self.split(0)
            self.plt_config('Three color channels', c_chs)
        if 'saveplt' in kwargs:
            plt.savefig('plot.png')
        plt.show()







if __name__ == '__main__':

    imgedit = ImageEditor('../resources/ponte.jpg')
    imgedit.crop((100, 200))
    imgedit.paint_all((0,0,255))
    imgedit.paint_squares((0,0,255))
    imgedit.flip('h')
    imgedit.rotate(30)
    imgedit.savefig(location='../exports')
    imgedit.plt_histogram(saveplt=True, color=True)
