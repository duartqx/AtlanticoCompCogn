import cv2
img = cv2.imread('../resources/ponte.jpg')
h, w = img.shape[:2]
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
print(type(M), type(M[0]), type(M[0][0]))
# <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.float64'>
