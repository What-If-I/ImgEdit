# coding=utf-8
import cv2 as cv

from my_opencv_funcs import (
    linear, negative, logimage, threshold, brightness, contrast_zoom, partlinear, pilaobraznoe, powimage)

image = cv.imread("../lion.jpg")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# линейного контрастирования полутонового изображения.
linear_img = linear(gray_image, 3.5, 255.0)
# преобразования полутонового изображения в негатив.
negative_img = negative(gray_image)
# логарифмического преобразования полутонового изображения.
logimage_img = logimage(gray_image, 40.0)
# степенного преобразования полутонового преобразования.
powimage_img = powimage(gray_image, 1.4, 1.1)
# кусочно-линейного преобразования изображения полутонового изображения.
partlinear_img = partlinear(gray_image, 2.0, 20.0)
# пороговой обработки полутонового изображения.
threshold_img = threshold(gray_image)
# преобразования яркостного среза полутонового изображения.
brightness_img = brightness(gray_image, 20.0, 100.0)
# контрастного масштабирования полутонового изображения.
contrast_zoom_img = contrast_zoom(gray_image)
# пилообразного контрастного масштабирования полутонового изображения
pilaobraznoe_img = pilaobraznoe(gray_image, 2)

cv.imshow("Original", image)
cv.imshow("gray_image", gray_image)
cv.imshow("Linear", linear_img)
cv.imshow("Negative", negative_img)
cv.imshow("Log", powimage_img)
cv.imshow("Pow", logimage_img)
cv.imshow("Threshold", threshold_img)
cv.imshow("Partlinear", partlinear_img)
cv.imshow("Brightness", brightness_img)
cv.imshow("Contrast zoom", contrast_zoom_img)
cv.imshow("Contrast saw", pilaobraznoe_img)

cv.waitKey(0)
cv.destroyAllWindows()
