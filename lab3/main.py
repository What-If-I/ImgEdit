import numpy as np
import cv2 as cv

from lab3.helpers import hist_lines_opencv, my_calc_hist, calc_hist
from matplotlib import pyplot as plt

image = cv.imread("../image.png")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
my_hist = my_calc_hist(gray_image)
eq_img = calc_hist(gray_image)
# hist = cv.calcHist(gray_image, 1, None, 255, (0, 255))
hist = cv.calcHist(gray_image, [1], None, [256], [0, 256])

# h = np.zeros((300, 256, 3))
# hist = np.int32(np.around(hist))
# for x, y in enumerate(hist):
#     cv.line(h, (x, 0), (x, y), (255, 255, 255))
# y = np.fliplr(h)


cv.imshow("Original", image)
cv.imshow("gist", hist_lines_opencv(gray_image))
# plt.hist(hist.ravel(), 256, [0, 256])
# plt.show()
cv.imshow("gray", gray_image)
cv.imshow("my_eq", eq_img)
cv.imshow("my_eq_hist", hist_lines_opencv(eq_img))



cv.waitKey(0)
cv.destroyAllWindows()
