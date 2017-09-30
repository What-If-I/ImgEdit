import cv2 as cv

from my_opencv_funcs.my_adaptive_threshold import my_adaptive_threshold
from my_opencv_funcs.threshold import my_threshold

image = cv.imread("../image.png")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


_retval, binary = cv.threshold(gray_image, 120, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
adaptive = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 10)
my_img = my_threshold(gray_image)
my_img2 = my_adaptive_threshold(gray_image, 51, 1.0)

cv.imshow("Original", image)
cv.imshow("gray", gray_image)
cv.imshow("binary", binary)
cv.imshow("my_binary", my_img)
cv.imshow("my_adaptive", my_img2)
cv.imshow("adaptive", adaptive)

cv.waitKey(0)
cv.destroyAllWindows()
