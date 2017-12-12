# coding=utf-8

import cv2 as cv
import numpy as np
import functions

img_path = "../image.png"

gray_image = cv.imread(img_path, 0)

dft_opencv = cv.dft(np.float32(gray_image), flags=cv.DFT_REAL_OUTPUT)
idft_opencv = cv.idft(dft_opencv)
dft_my = functions.fft2(gray_image)
idft_my = np.uint8(map(functions.idft, dft_my))

dct_opencv = cv.dct(np.float32(gray_image))
idct_opencv = cv.idct(dct_opencv)
dct_opencv = np.uint8(dct_opencv)  # convert back
idct_opencv = np.uint8(idct_opencv)

dct_my = map(functions.dct, (np.float32(gray_image)))
idct_my = map(functions.dct, dct_my)
dct_my = np.uint8(idct_my)
idct_my = np.uint8(dct_my)

cv.imshow("gray", gray_image)
cv.imshow("dft_opencv", dft_opencv)
cv.imshow("idft_opencv", idft_opencv)
cv.imshow("idft_my", idft_my)
cv.imshow("dct_opencv", dct_opencv)
cv.imshow("idct_opencv", idct_opencv)
cv.imshow("dct_my", dct_my)
cv.imshow("idct_my", idct_my)

cv.waitKey(0)
cv.destroyAllWindows()
