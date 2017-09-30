import numpy as np

from helpers import count_pixels


def calc_delta(lst):
    max_val = max(lst)

    if lst.count(max_val) != 1:
        indices = [i for i, x in enumerate(lst) if x == max_val]
        return int(sum(indices) / len(lst))

    return lst.index(max_val)


def calc_osu_threshold(img_array):
    counted_pixels_by_brightness = count_pixels(img_array)

    N = float(img_array.size)
    Pi = [counted_pixels_by_brightness.get(n, 0.0) / N for n in range(256)]
    P1 = [sum(Pi[0:k + 1]) for k in range(256)]

    M = []
    for k in range(256):
        el = 0
        for i in range(k + 1):
            el += i * Pi[i]
        M.append(el)

    Mg = 0.0
    for i in range(256):
        Mg += i * Pi[i]

    d2b = []
    for k in range(256):
        P1k = float(P1[k])
        if P1k != 0:
            try:
                res = pow(Mg * P1k - M[k], 2) / (P1k * (1 - P1k))
            except ZeroDivisionError:
                res = 0.0
        else:
            res = 0.0
        d2b.append(res)

    delta = calc_delta(d2b)

    # Not nessecary
    d2g = 0.0
    for i in range(256):
        d2g += pow(i - Mg, 2) * Pi[i]

    n_delta = d2b[delta] / d2g

    return delta


def my_threshold(image_array, thresh=None):

    if not thresh:
        thresh = calc_osu_threshold(image_array)

    MAX_VAL = np.uint8(255)
    MIN_VAl = np.uint8(0)
    new_img = image_array.copy()

    for x, row in enumerate(image_array):
        for y, p in enumerate(row):
            new_img[x, y] = MAX_VAL if p >= thresh else MIN_VAl

    return new_img
