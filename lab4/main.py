# coding=utf-8

import cv2 as cv

import functions as my_funcs

img_path = "../image.png"

gray_image = cv.imread(img_path, flags=0)

averaged_img = my_funcs.average_blur_filter(gray_image, 3)
opencv_averaged_img = cv.blur(gray_image, (6, 6))
gausian_blur_img = my_funcs.gausian_blur_filter(gray_image, 5, 1.0)
opencv_gausian_blur_img = cv.GaussianBlur(gray_image, (5, 5), 1)
median_blur_filter_img = my_funcs.median_blur_filter(gray_image, 7)
opencv_median_blur_filter_img = cv.medianBlur(gray_image, 7)
biliteral_filter_img = my_funcs.billateral_blur_filer(gray_image, 9, 75, 75)
opencv_biliteral_filter_img = cv.bilateralFilter(gray_image, 9, 75, 75)
roberts_img = my_funcs.roberts_method(gray_image)
sobel_method = my_funcs.sobel_method(gray_image)
sobel_method_opencv = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3)
prewitt_method = my_funcs.prewitt_method(gray_image)
laplasian_method = my_funcs.laplasian_method(gray_image)
log_filter = my_funcs.log_filter(gray_image)
dog_filter = my_funcs.dog_filter(gray_image, 3.0, 5.5)
sharpness_filter = my_funcs.sharpness_filter(gray_image)

cv.imshow("gray", gray_image)
cv.imshow("averaged", averaged_img)
cv.imshow("opencv_averaged_img", opencv_averaged_img)
cv.imshow("gausian_blur_img", gausian_blur_img)
cv.imshow("Opencv_gausian_blur_img", opencv_gausian_blur_img)
cv.imshow("median_blur_filter_img", median_blur_filter_img)
cv.imshow("opencv_median_blur_filter_img", opencv_median_blur_filter_img)
cv.imshow("biliteral_filter_img", biliteral_filter_img)
cv.imshow("opencv_biliteral_filter_img", opencv_biliteral_filter_img)
cv.imshow("roberts_img", roberts_img)
cv.imshow("sobel_method", sobel_method)
cv.imshow("sobel_method_opencv", sobel_method_opencv)
cv.imshow("previtt_method", prewitt_method)
cv.imshow("laplasian_method", laplasian_method)
cv.imshow("log_filter", log_filter)
cv.imshow("dog_filter", dog_filter)
cv.imshow("sharpness_filter", sharpness_filter)

cv.waitKey(0)
cv.destroyAllWindows()
