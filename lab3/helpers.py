import cv2 as cv
import numpy as np


def hist_lines_opencv(im):
    h = np.zeros((300, 256, 3))
    hist_item = cv.calcHist([im], [0], None, [256], [0, 256])
    # cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    for x, y in enumerate(hist):
        cv.line(h, (x, 0), (x, y), (255, 255, 255))
    y = np.flipud(h)
    return y
