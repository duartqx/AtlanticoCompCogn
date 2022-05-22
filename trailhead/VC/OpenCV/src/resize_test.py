import numpy as np
import cv2

img = cv2.imread('../resources/ponte.jpg')
h: int; w: int
h, w, _ = img.shape
print(h, w)
print(type(h), type(w))
