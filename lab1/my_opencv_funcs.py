import numpy as np


def linear(img_arr, c, max_value):
    res_img = img_arr.copy()
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            new_pix = p * c
            res_img[x, y] = new_pix if new_pix <= max_value else max_value
    return res_img


def negative(img_arr):
    res_img = img_arr.copy()
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            res_img[x, y] = 255.0 - p
    return res_img


def logimage(img_arr, c):
    res_img = img_arr.copy()
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            res_img[x, y] = c * np.log(1 + p)
    return res_img


def powimage(img_arr, c, yi):
    """
    :param img_arr: image numpy array
    :param c: const
    :param yi: power
    """
    res_img = img_arr.copy()
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            new_p = c * pow(p, yi)
            res_img[x, y] = new_p if new_p < 255 else 255
    return res_img


def partlinear(img_arr, c, b):
    res_img = img_arr.copy()
    max_val = 255.0
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            new_pix = c * p + b
            res_img[x, y] = new_pix if new_pix <= max_val else max_val
    return res_img


def threshold(img_arr):
    res_img = img_arr.copy()
    d = 255.0 / 2.0
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            res_img[x, y] = 255 if p < d else 0
    return res_img


def brightness(img_arr, d_min, d_max):
    gmax, gmin = 255.0, 0.0
    res_img = img_arr.copy()
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            res_img[x, y] = gmax if d_min <= p < d_max else gmin
    return res_img


def contrast_zoom(img_arr):
    gmax, gmin = 255.0, 0.0
    f_min, f_max = 100.0, 200.0
    res_img = img_arr.copy()
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            a = (gmax - gmin) / (f_max - f_min)
            b = ((gmin * f_max) - (gmax * f_min)) / (f_max - f_min)
            s = a * p + b
            res_img[x, y] = s if s <= gmax else gmax
    return res_img


def pilaobraznoe(img_arr, zones=2):
    f_min, f_max = [], []
    g_min, g_max = 0.0, 255.0
    summ = 0.0
    step = g_max / zones

    while summ < g_max:
        f_min.append(summ)
        summ += step
        f_max.append(summ)

    res_img = img_arr.copy()
    for x, row in enumerate(res_img):
        for y, p in enumerate(row):
            for i in range(zones):
                a = (g_max - g_min) / (f_max[i] - f_min[i])
                b = ((g_min * f_max[i]) - (g_max * f_min[i])) / (f_max[i] - f_min[i])
                s = a * p + b
                res_img[x, y] = s if s <= g_max else g_max
    return res_img
