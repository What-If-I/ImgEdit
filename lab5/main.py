# coding=utf-8

import cv2 as cv


img_path = "../image.png"

gray_image = cv.imread(img_path, flags=0)

cv.imshow("gray", gray_image)


cv.waitKey(0)
cv.destroyAllWindows()
