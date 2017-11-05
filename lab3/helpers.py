from collections import Counter

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


def count_pixels(img_array):
    total_count = Counter()
    for row in img_array:
        total_count = total_count + Counter(row)
    return total_count


def my_calc_hist(img_array):
    counted_pixels_by_brightness = count_pixels(img_array)
    max, min = img_array.max, img_array.min
    total_amount_of_pixels = float(img_array.size)
    pixel_probabilities = {
        brightness: amount / total_amount_of_pixels for brightness, amount in counted_pixels_by_brightness.items()
    }
    probs = []
    for i in range(256):
        probs.append(pixel_probabilities.get(i, 0))
    return probs


def calc_hist(img_array):
    max_, min_ = img_array.max(), img_array.min()
    total_amount_of_pixels = float(img_array.size)
    counted_pixels_by_brightness = count_pixels(img_array)
    probs = my_calc_hist(img_array)
    for i, prob in enumerate(probs):
        probs[i] = probs[i-1] + prob
    new_img = img_array.copy()
    for x, row in enumerate(new_img):
        for y, pix in enumerate(row):
            res = ((max_ - min_) * probs[pix] + min_)
            assert res <= 255
            new_img[x, y] = res
    return new_img
