import cv2 as cv

from functions import average_blur_filter


image = cv.imread("../image.png")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
averaged_img = average_blur_filter(gray_image, 3)


cv.imshow("Original", image)
cv.imshow("gray", gray_image)
cv.imshow("averaged", averaged_img)


cv.waitKey(0)
cv.destroyAllWindows()
