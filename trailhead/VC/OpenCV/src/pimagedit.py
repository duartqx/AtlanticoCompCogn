from queue import Queue
from typing import Callable, TypeAlias, Union
from os import path
import cv2                                                      # type: ignore
import mahotas as mh                                            # type: ignore
import matplotlib.pyplot as plt                                 # type: ignore
import numpy as np                                              # type: ignore

BGRTuple: TypeAlias = tuple[np.uint8, np.uint8, np.uint8]
CVColorImage: TypeAlias = 'np.ndarray[np.ndarray[np.ndarray[np.uint8]]]'
CVBwImage: TypeAlias = 'np.ndarray[np.ndarray[np.uint8]]'
CVImage: TypeAlias = Union[CVBwImage, CVColorImage]
CVColorChannels: TypeAlias = tuple[CVBwImage, ...]
# CVColorChannels is a tuple with no defined number of CVBwImage in it
# Could be a tuple[CVBwImage, CVBwImage, CVBwImage] or tuple[CVBwImage]
ObjectEdge: TypeAlias = 'tuple[np.ndarray[np.ndarray[np.ndarray[np.int32]]]]'
RotationMatrix: TypeAlias = 'np.ndarray[np.ndarray[np.float64]]'

class Simp:
    ''' Simple Image Manipulator in Python (Simp) using OpenCV '''
    def __init__(self, img: str, savelocation: str, verb: bool=False) -> None:
        # Save location for edited images and Plots
        self.savelocation = savelocation
        self.name_n_ext: tuple[str, str] = path.splitext(path.basename(img))
        # Reads the image
        self.__img: CVImage = cv2.imread(img)
        self.__bw: CVBwImage = self._grayscale()
        # Initializes last_edited variable to None and starts a Queue
        # last_edited is stored so that we can return it with the method show()
        # whitout having to get it from the Queue
        self.last_edited: CVImage = None
        self.edited_imgs: Queue[CVImage] = Queue()
        self.verb = verb
        # if verb then some methods will print messages (verbose) when called

    @property
    def img(self) -> CVImage:
        return self.__img

    @property
    def bw(self) -> CVBwImage:
        return self.__bw

    def __len__(self) -> int:
        return self.edited_imgs.qsize()

    def __repr__(self) -> str:
        return f'''Editing {self.img}
        You have {self.edited_imgs.qsize()} unsaved edited images'''

    def _put_last(self, *args, operation: str='') -> None:
        ''' Adds edted_img to self.edited_imgs Queue and sets
        self.last_edited to edted_img '''
        for edted_img in args:
            self.edited_imgs.put((operation, edted_img))
            self.last_edited = edted_img
        if self.verb:
            print(f'{operation} image file and added it to the save Queue')

    @staticmethod
    def _savefig(flname: str, fig: CVImage) -> None:
        cv2.imwrite(flname, fig)

    def _get_filename(self, i: int, operation: str) -> str:
        ''' Builds the filename string to be used with savefig '''
        return f'{self.name_n_ext[0]}_ed_{i}_{operation}{self.name_n_ext[1]}'

    def savefigs(self) -> None:
        ''' Saves the edited image to an image file '''
        for i in range(1, self.edited_imgs.qsize()+1):
            img_tup: tuple[str, CVImage] = self.edited_imgs.get()
            fn: str = self._get_filename(i, operation=img_tup[0])
            flname_path: str = path.join(self.savelocation, fn)
            self._savefig(flname_path, img_tup[1])

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
        return edted_img

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
        self._put_last(altered_img, operation='Altered_color_mode_from')
        return altered_img

    def _grayscale(self) -> CVBwImage:
        ''' Converts and returns self.img to grayscale '''
        return cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2GRAY)

    def grayscale(self, q: bool=True) -> CVBwImage:
        ''' Converts and returns self.img as a black and white image 
        Arg
            q (bool) if True saves grayscale to the Queue
        '''
        self._put_last(self.bw, operation='Grayscaled')
        return self.bw

    def split(self, *args) -> tuple[CVBwImage, CVBwImage, CVBwImage]:
        ''' Splits self.img color channels and adds them to the save Queue'''
        blue_ch: CVImage; green_ch: CVImage; red_ch: CVImage;
        blue_ch, green_ch, red_ch = cv2.split(self.img)
        if not args:
            self._put_last(blue_ch, green_ch, red_ch, operation='Splitted')
        return blue_ch, green_ch, red_ch

    @staticmethod
    def merge(b_ch: CVBwImage, 
              g_ch: CVBwImage, 
              r_ch: CVBwImage, 
              savefig=False) -> CVImage:
        ''' Merges three separated color channels into a single image file '''
        merged_chs: CVImage = cv2.merge([b_ch, g_ch, r_ch])
        if savefig:
            Simp._savefig('merged_img.png', merged_chs)
        return merged_chs

    @staticmethod
    def merge_single_channel(color_ch: CVBwImage, 
                             color: str, 
                             savefig: bool=False) -> CVImage:
        zeros_ch: CVBwImage = np.zeros(color_ch.shape[:2], dtype='uint8')
        merged_img: CVImage
        if color == 'red':
            merged_img = cv2.merge([zeros_ch, zeros_ch, color_ch])
        elif color == 'green':
            merged_img = cv2.merge([zeros_ch, color_ch, zeros_ch])
        elif color == 'blue':
            merged_img = cv2.merge([color_ch, zeros_ch, zeros_ch])
        if savefig:
            Simp._savefig('merged_single_channel.png', merged_img)
        return merged_img

    @staticmethod
    def _plt_config(method: str, c_chnls: CVColorChannels) -> None:
        ''' plt config operations that is called when using the plt_histogram()
        method '''
        plt.figure()
        plt.title(f'Histogram {method}')
        plt.xlabel('Intensity')
        plt.ylabel('Qtty of Pixels')
        colors: list[str]
        if len(c_chnls) == 1: colors = ['gray']
        else: colors = ['b','g','r']
        for c_ch, color in zip(c_chnls, colors):
            hist: 'np.ndarray[np.float32]'= Simp._calc_hist(c_ch)
            plt.plot(hist, color=color)
            plt.xlim([0, 256])

    @staticmethod
    def _calc_hist(img: CVImage) -> 'np.ndarray[np.float32]':
        ''' Returns the np array with the img histogram '''
        return cv2.calcHist([img], [0], None, [256], [0, 256])

    def equalize(self, img: CVBwImage=None) -> CVImage:
        ''' Equalizes img's histogram adds it to the save Queue and returns the
        img so that it's histogram can be plotted '''
        if img is None:
            img = self.bw
        eq_img: CVImage = cv2.equalizeHist(img)
        self._put_last(eq_img, operation='Equalized')
        return eq_img

    @staticmethod
    def _ravel(g_img: CVBwImage) -> None:
        ''' Plots ravel histogram '''
        plt.hist(g_img.ravel(), 256, [0, 256])

    def plt_histogram(self, **kwargs) -> None:
        '''
        Plots self.img's histogram
        **kwargs:
            color: bool
            equilize: bool
            ravel: bool
            saveplt: bool
        '''
        if kwargs.get('color', None) is not None:
            # Plots three histogram for each color channel
            c_chs: CVColorChannels = self.split(0)
            self._plt_config('Three color channels', c_chs)
        else:
            # Plot the single B&W histogram
            if kwargs.get('equalize', None) is not None:
                g_img = self.equalize(self.bw)
                self._ravel(g_img)
            elif kwargs.get('ravel', None) is not None:
                # Plots ravel histogram
                self._ravel(self.bw)
            else:
                # Plots line histogram for b&w image
                self._plt_config('B&W', (self.bw,)) 
        if kwargs.get('saveplt', None) is not None:
            plt.savefig(path.join(self.savelocation, 'plot.png'))
        plt.show()

    @staticmethod
    def _blur(method: Callable, img: CVImage, amount: int) -> CVImage:
        ''' Applies gaussian blur to self.img '''
        return method(img, (amount, amount), 0)

    def gaussian(self, amount: int) -> CVImage:
        ''' Applies gaussian blur to self.img by 'amount' pixels '''
        blurred_img: CVImage = self._blur(cv2.GaussianBlur, self.img, amount)
        self._put_last(blurred_img, operation='Gaussian_blurred')
        return blurred_img

    def _blur_grid(self, func: Callable, oper: str, **kwargs) -> CVImage:
        '''
        Create Blurs grid of self.img with cv2.medianBlur, cv2.blur or
        cv2.bilateralFilter
        Args:
            func (Callable) Either cv2.medianBlur or cv2.blur
            oper (str) either 'Median Blurred' or 'Mean Blurred'
            kwargs
                kwargs['3'] (int | tuple[int]): (3,), (3, 3) or (3, 21, 21)
                kwargs['5'] (int | tuple[int]): (5,), (5, 5) or (5, 35, 35)
                kwargs['7'] (int | tuple[int]): (7,), (7, 7) or (7, 49, 49)
                kwargs['9'] (int | tuple[int]): (9,), (9, 9) or (9, 63, 63)
                kwargs['11'] (int | tuple[int]): (11,), (11, 11) or (11, 77, 77)
        '''
        img: CVImage = self.img.copy()[::2, ::2]
        edted_img: CVImage = np.vstack([
            np.hstack([img, func(img, *kwargs['3'])]),
            np.hstack([func(img, *kwargs['5']), func(img, *kwargs['7'])]),
            np.hstack([func(img, *kwargs['9']), func(img, *kwargs['11'])]) ])
        self._put_last(edted_img, operation=oper)
        return edted_img

    def mean_blur_grid(self) -> CVImage:
        ''' Mean blur grid '''
        kwargs: dict[str, tuple[tuple[int, int]]]
        kwargs = {str(i): ((i, i),) for i in range(3, 12, 2)}
        return self._blur_grid(cv2.blur, 'Mean_Blurred', **kwargs)

    def median_blur_grid(self) -> CVImage:
        ''' Median blur grid '''
        kwargs: dict[str, tuple[int]] = {str(i): (i,) for i in range(3, 12, 2)}
        return self._blur_grid(cv2.medianBlur, 'Median_Blurred', **kwargs)

    def bilateral_filter_grid(self) -> CVImage:
        ''' Removes image noise while trying to preserve edges by calculating a
        bilateral filter '''
        kwargs: dict[str, tuple[int, int, int]]
        kwargs = {str(i): (i, i*7, i*7) for i in range(3, 12, 2)}
        return self._blur_grid(cv2.bilateralFilter, 'Bilateral', **kwargs)

    def _four_grid(self, *args) -> CVImage: 
        '''
        Returns a grid of four images
        args:
            img_1: CVImage
            img_2: CVImage
            img_3: CVImage
            img_4: CVImage
        '''
        first_row: CVImage = np.hstack([args[0], args[1]])
        second_row: CVImage = np.hstack([args[2], args[3]])
        return np.vstack([first_row, second_row]) 

    @staticmethod
    def _get_thresholds(img: CVBwImage) -> tuple[CVBwImage, CVBwImage]:
        ''' Returns binary threshold and binary inv threshold images '''
        T: float; bin: CVImage; binI: CVImage
        T, bin = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
        T, binI = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
        return bin, binI

    @staticmethod
    def _get_adap(img: CVBwImage, method: str) -> CVBwImage:
        ''' Returns a cv2.adaptiveThreshold filtered CVImage '''
        bin: int; adap: int
        bin = cv2.THRESH_BINARY_INV
        if method == 'mean':
            adap = cv2.ADAPTIVE_THRESH_MEAN_C
        elif method == 'gaussian':
            adap = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        return cv2.adaptiveThreshold(img, 255, adap, bin, 21, 5)

    def _get_adp_thresholds(self, 
                               img: CVBwImage) -> tuple[CVBwImage, CVBwImage]:
        bin_mean: CVBwImage; bin_gaussian: CVBwImage
        bin_mean = self._get_adap(img, method='mean')
        bin_gaussian = self._get_adap(img, method='gaussian')
        return bin_mean, bin_gaussian

    def _threshold_grid(self, adap: bool=False) -> CVBwImage:
        ''' Returns a grid with four binary threshold edits 
        normal threshold if adap = False and adaptiveThreshold if adap = True
        '''
        blurred: CVBwImage = self._blur(cv2.GaussianBlur, self.bw, amount=7)

        bin_a: CVBwImage; bin_b: CVBwImage; 

        if adap:
            bin_a, bin_b = self._get_adp_thresholds(blurred)
        else:
            bin_a, bin_b = self._get_thresholds(blurred)
            #bitand: CVImage = cv2.bitwise_and(img, img, mask=bin_b)
        # Get the grid
        bin_grid: CVImage
        bin_grid = self._four_grid(self.bw, blurred, bin_a, bin_b)
        self._put_last(bin_grid)
        return bin_grid

    def normal_binary_threshold_grid(self) -> CVBwImage:
        ''' Normal binary threshold filter grid '''
        return self._threshold_grid()

    def adaptive_binary_threshold_grid(self) -> CVBwImage:
        ''' Adaptive binary threshold filter grid '''
        return self._threshold_grid(adap=True)

    def _get_normalized(self, T, img: CVImage=None) -> CVImage:
        ''' Normalizes image '''
        temp: CVImage
        if img is not None:
            temp = img.copy()
        else:
            temp = self.bw.copy()
        temp[temp > T] = 255
        temp[temp < 255] = 0
        return cv2.bitwise_not(temp)

    def _get_mahota(self, model: Callable, blrrd: CVBwImage, b: bool=False) -> CVBwImage:
        ''' Returns a normalized mahota otsu or rc 
        Arg:
            model (Callable): Can be either mh.thresholding.otsu or
            mh.thresholding.rc
            blurred (CVBwImage): Black and white self.img with gaussian applied
        '''
        if b:
            # pass blrrd to self._get_normalized so that when identifying
            # objects we search with the blurred image instead of of the normal
            # self.bw
            return self._get_normalized(model(blrrd), blrrd)
        return self._get_normalized(model(blrrd))

    def mahotas_grid(self) -> CVBwImage:
        ''' Returns a grid with mahota's thresholding using otsu's and rc's
        algorithms '''
        blrrd: CVBwImage = self._blur(cv2.GaussianBlur, self.bw, 7)
        otsu: CVBwImage = self._get_mahota(mh.thresholding.otsu, blrrd)
        rc: CVBwImage = self._get_mahota(mh.thresholding.rc, blrrd)
        mh_grid: CVBwImage = self._four_grid(self.bw, blrrd, otsu, rc)
        self._put_last(mh_grid, operation='Otsu_RC_Threshold')
        return mh_grid

    @staticmethod
    def _get_sobel(img: CVImage, axis: tuple[int, int]) -> CVBwImage:
        ''' Calculates and returns Sobel '''
        return np.uint8(np.absolute(cv2.Sobel(img, cv2.CV_64F, *axis)))

    def sobel(self) -> CVBwImage:
        ''' Returns a grid with edge detected images using the sobel
        algorithm '''
        _x: CVBwImage = self._get_sobel(self.bw, (1,0))
        _y: CVBwImage = self._get_sobel(self.bw, (0,1))
        sobel: CVBwImage = cv2.bitwise_or(_x, _y)
        sobel_grid: CVBwImage = self._four_grid(self.bw, _x, _y, sobel)
        self._put_last(sobel_grid, operation='Sobel')
        return sobel_grid

    def _get_lap(self) -> CVBwImage:
        ''' Returns the laplacian filtered self.bw '''
        return np.uint8(np.absolute(cv2.Laplacian(self.bw, cv2.CV_64F)))

    def laplace(self) -> CVBwImage:
        ''' Returns a vertical stack of the original self.bw + laplacian edge
        detection applied '''
        lap_grid: CVBwImage = np.vstack([self.bw, self._get_lap()])
        self._put_last(lap_grid, operation='Laplace')
        return lap_grid

    @staticmethod
    def _both_canny(blurred: CVBwImage) -> tuple[CVBwImage, CVBwImage]:
        ''' Returns two canny edge detected images '''
        return cv2.Canny(blurred, 20, 120), cv2.Canny(blurred, 70, 200)

    def canny(self) -> CVBwImage:
        ''' Builds and returns a _four_grid montage with canny's edge detected
        images '''
        blrrd: CVBwImage = self._blur(cv2.GaussianBlur, self.bw, amount=7)
        canny_1: CVBwImage; canny_2: CVBwImage; canny_grid: CVBwImage
        canny_1, canny_2 = self._both_canny(blrrd)
        canny_grid = self._four_grid(self.bw, blrrd, canny_1, canny_2)
        self._put_last(canny_grid, operation='Canny')
        return canny_grid

    @staticmethod
    def write_text(img: CVImage, 
                          text: str, color: BGRTuple=(255,0,0)) -> CVImage:
        ''' Writes text to the img so that we can create titles or add
        information of what the filter did to the img '''
        fnt: int = cv2.FONT_HERSHEY_SIMPLEX
        return cv2.putText(img, text, (10,20), fnt, 0.5, color, 0, cv2.LINE_AA)

    def write_identified(self, f_cvimgs: tuple[CVImage]) -> CVImage:
        ''' Writes the title of all four images to be returned as a
        _four_grid '''
        bwimg = self.write_text(f_cvimgs[0].copy(), 'BW image', 0)
        blrrd = self.write_text(f_cvimgs[1].copy(), 'Blurred Image', 0)
        otsu = self.write_text(f_cvimgs[2].copy(), 
                               'Binary image with Otsu\'s method', 255)
        edges = self.write_text(f_cvimgs[3].copy(), 
                                'Canny\'s edge detected', 255)
        return self._four_grid(bwimg, blrrd, otsu, edges)

    def _draw_contour(self, objects: ObjectEdge) -> CVImage:
        ''' Draws identified objects on self.img with cv2.drawContours '''
        t: str = f'{str(len(objects))} objects found!'
        return self.write_text(
               cv2.drawContours(self.img.copy(), objects, -1, (255,0,0), 2), t)

    def identify(self) -> CVImage:
        ''' Identify objects in an image using cv2.findContours and draws their
        countour with cv2.drawContours '''
        blrrd: CVImage = self._blur(cv2.blur, self.bw, 7)
        otsu: CVBwImage = self._get_mahota(mh.thresholding.otsu, blrrd, b=True)
        edges: CVBwImage = cv2.Canny(otsu, 70, 150)

        # Counting elements by edges
        object: ObjectEdge
        objects, _ = cv2.findContours(edges.copy(), 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        f_cvimgs: tuple[CVImage] = (self.bw, blrrd, otsu, edges)
        idt_grid: CVImage = self.write_identified(f_cvimgs)
        idt_ovr_org: CVImage = self._draw_contour(objects)

        self._put_last(idt_grid, idt_ovr_org, operation='Identifying')
        return idt_grid, idt_ovr_org
