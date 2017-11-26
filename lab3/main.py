import cv2 as cv

from lab3.helpers import hist_lines_opencv, my_calc_hist, calc_hist
# from matplotlib import pyplot as plt

image = cv.imread("../image.png")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
my_hist = my_calc_hist(gray_image)
eq_img = calc_hist(gray_image)
hist = cv.calcHist(gray_image, [1], None, [256], [0, 256])


cv.imshow("Original", image)
cv.imshow("gist", hist_lines_opencv(gray_image))
# plt.hist(hist.ravel(), 256, [0, 256])
# plt.show()
cv.imshow("gray", gray_image)
cv.imshow("equlized", eq_img)
cv.imshow("equlized_hist", hist_lines_opencv(eq_img))



cv.waitKey(0)
cv.destroyAllWindows()
