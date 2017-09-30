import cv2 as cv

from my_opencv_funcs import (
    linear, negative, logimage, threshold, brightness, contrast_zoom, partlinear, pilaobraznoe, powimage)

image = cv.imread("../image.png")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

linear_img = linear(gray_image, 3.5, 255.0)
negative_img = negative(gray_image)
logimage_img = logimage(gray_image, 50.0)
partlinear_img = partlinear(gray_image, 30.0, 20.0)
threshold_img = threshold(gray_image)
brightness_img = brightness(gray_image, 30.0, 120.0)
contrast_zoom_img = contrast_zoom(gray_image)
pilaobraznoe_img = pilaobraznoe(gray_image, 2)
powimage_img = powimage(gray_image, 10.0, 2)

cv.imshow("Original", image)
cv.imshow("gray_image", gray_image)
cv.imshow("linear_img", linear_img)
cv.imshow("negative_img", negative_img)
cv.imshow("logimage_img", logimage_img)
cv.imshow("partlinear_img", partlinear_img)
cv.imshow("brightness_img", brightness_img)
cv.imshow("contrast_zoom_img", contrast_zoom_img)
cv.imshow("pilaobraznoe_img", pilaobraznoe_img)
cv.imshow("powimage_img", powimage_img)

cv.waitKey(0)
cv.destroyAllWindows()
