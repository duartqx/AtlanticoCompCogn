from __future__ import annotations
from numpy import ndarray, uint8
from queue import Queue
from typing import Optional, TypeAlias
import cv2

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

    def _put_last(self, edted_img: CVImage) -> None:
        ''' Adds edted_img to self.edited_imgs Queue and sets
        self.last_edited to edted_img '''
        self.edited_imgs.put(edted_img)
        self.last_edited = edted_img

    def savefig(self, location: str=None) -> None:
        ''' Saves the edited image to an image file '''
        if location is not None:
            assert isinstance(location, str), 'Directory must be a string'
            if not location.endswith('/'):
                location += '/'
        else:
            location = ''

        for i in range(self.edited_imgs.qsize()):
            filename: str = f'{location}{self.img_name}_ed_{i+1}.{self.img_ext}'
            cv2.imwrite(filename, self.edited_imgs.get())
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
        self._put_last(edted_img)
        if self.verb:
            print('Painted the image file and added it to the save Queue')
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
        self._put_last(cropped_img)
        if self.verb:
            print('Cropped the image file and added it to the save Queue')
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
        self._put_last(resized_img)
        if self.verb:
            print('Resized the image file and added it to the save Queue')
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
        self._put_last(flipped_img)
        if self.verb:
            print('Flipped the image file and added it to the save Queue')
        return flipped_img

    def rotate(self, d: int) -> CVImage:
        ''' Rotates the image related to the center by (d) degrees '''
        h: int; w: int
        h, w = self.img.shape[:2] # Grabs width and height of self.img
        center: tuple[int, int] = (w // 2, h // 2) # Finds the center
        rot_matrix: RotationMatrix = cv2.getRotationMatrix2D(center, d, 1.0)
        rotated_img: CVImage = cv2.warpAffine(self.img, rot_matrix, (w, h))
        self._put_last(rotated_img)
        if self.verb:
            print('Rotated the image file and added it to the save Queue')
        return rotated_img




if __name__ == '__main__':

    imgedit = ImageEditor('../resources/ponte.jpg')
    imgedit.crop((100, 200))
    imgedit.paint_all((0,0,255))
    imgedit.paint_squares((0,0,255))
    imgedit.flip('h')
    imgedit.rotate(30)
    imgedit.savefig(location='../exports')

