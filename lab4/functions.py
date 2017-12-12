from math import exp, sqrt, pow, pi
import cv2 as cv
import numpy as np


def iter_mask(x, y, threshold):
    mask_width = x - threshold, x + threshold + 1
    mask_height = y - threshold, y + threshold + 1  # Get rid of that +1? naaaah
    for mask_x in xrange(*mask_width):
        for mask_y in xrange(*mask_height):
            yield mask_x, mask_y


def apply_mask(mask, image, coords):
    x, y = map(int, coords)
    new_val = 0
    max_width, max_height = image.shape
    mask_len = len(mask)
    mask_center = int(mask_len / 2)
    for mask_x, row in enumerate(mask):
        offset_x = mask_x - mask_center
        get_x = offset_x + x
        if not 0 <= get_x < max_width:
            continue  # skip non-existing indexes
        for mask_y, mask_val in enumerate(row):
            offset_y = mask_y - mask_center
            get_y = offset_y + y
            if not 0 <= get_y < max_height:
                continue  # skip non-existing indexes
            pix = int(image[get_x][get_y])
            new_val += pix * mask_val / mask_len ** 2

    return new_val


def average_blur_filter(image_array, n):
    assert n > 1

    result_img = image_array.copy()
    threshold = n - 1
    max_x, max_y = result_img.shape

    for x, row in enumerate(image_array):
        for y, p in enumerate(row):

            mask_pixels = []
            for mask_x, mask_y in iter_mask(x, y, threshold):
                if not (0 <= mask_x < max_x) or not (0 <= mask_y < max_y):
                    continue  # skip non-existing indexes
                pix = result_img[mask_x][mask_y]
                mask_pixels.append(pix)

            avg = sum(mask_pixels) / len(mask_pixels)
            result_img[x][y] = avg

    return result_img


def calc_gaussian_mask(x, y, sigma):
    return exp(-(pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2))) / \
           (2 * pi * pow(sigma, 2))


def calc_distance(x, y, i, j):
    return sqrt(pow(x - i, 2)) + pow(y - j, 2)


def gausian_blur_filter(image_array, n, sigma):
    assert n > 1

    result_img = image_array.copy()
    threshold = n - 1
    max_x, max_y = result_img.shape

    for x, row in enumerate(image_array):
        for y, p in enumerate(row):
            result_pix = 0
            for mask_x, mask_y in iter_mask(x, y, threshold):
                if not (0 <= mask_x < max_x) or not (0 <= mask_y < max_y):
                    continue  # skip non-existing indexes
                pix = result_img[mask_x][mask_y]
                g = calc_gaussian_mask(mask_x - x, mask_y - y, sigma)
                result_pix += pix * g

            result_img[x][y] = result_pix

    return result_img


def median_blur_filter(image_array, diameter):
    result_img = image_array.copy()
    width_range, height_range = result_img.shape

    for x, row in enumerate(image_array):
        for y, p in enumerate(row):
            window = []
            n = 0
            i = -diameter if y >= diameter else -y
            d1 = diameter if y + diameter <= height_range else height_range - y

            for _ in xrange(d1):
                i += 1
                j = -diameter if x >= diameter else -x
                d2 = diameter if x + diameter <= width_range else width_range - x

                for _ in xrange(d2):
                    j += 1
                    window.append(image_array[x + j, y + i])
                    n += 1

            window.sort()
            new_pix = window[n / 2]
            result_img[x][y] = new_pix

    return result_img


def bilateral_filter(img_arr, x, y, diameter, sigma_i, sigma_s):
    res = 0
    w_p = 0
    half = float(diameter) / 2
    max_x, max_y = map(lambda pos: pos - 1, img_arr.shape)
    for i in xrange(diameter):
        neighbor_y = int(y - (half - i))
        for j in xrange(diameter):
            neighbor_x = int(x - (half - j))
            if neighbor_x > max_x:
                neighbor_x = max_x
            if neighbor_y > max_y:
                neighbor_y = max_y

            pix = img_arr[x][y]
            pix_neighbor = img_arr[neighbor_x][neighbor_y]

            gi = calc_gaussian_mask(int(pix_neighbor) - int(pix), 0, sigma_i)
            gs = calc_gaussian_mask(
                calc_distance(x, y, neighbor_x, neighbor_y), 0, sigma_s)

            w = gi * gs
            res += pix_neighbor * w
            w_p += w

    return int(res / w_p)


def billateral_blur_filer(image_array, diameter, sigma_i, sigma_s):
    max_width, max_height = image_array.shape

    result_img = image_array.copy()
    for x, row in enumerate(image_array):
        for y, p in enumerate(row):
            if any((x > diameter * 2, y > diameter * 2, y < max_height, x < max_width)):
                new_res = bilateral_filter(image_array, x, y, diameter, sigma_i, sigma_s)
                result_img[x][y] = new_res

    return result_img


def roberts_method(image_array):
    result_img = image_array.copy()
    max_width, max_height = image_array.shape

    for x, row in enumerate(image_array):
        for y, p in enumerate(row):
            y1 = y + 1 if y + 1 < max_height else y
            x1 = x + 1 if x + 1 < max_height else x
            py1 = int(image_array[x][y1])
            px1 = int(image_array[x1][y])
            pxy1 = int(image_array[x1][y1])

            g1 = int(p) - pxy1
            g2 = px1 - py1
            new_pix = g1 * g2 + g2 * g2
            result_img[x][y] = int(sqrt(new_pix)) if new_pix > 0 else 0

    return result_img


def sobel_method(img_array):
    result_img = img_array.copy()
    mask = (
        (-1, 0, 1),
        (-2, 0, 2),
        (-1, 0, 1),
    )
    for x, row in enumerate(img_array):
        for y, p in enumerate(row):
            new_pix = apply_mask(mask, img_array, coords=(x, y))
            result_img[x][y] = new_pix

    return result_img


def prewitt_method(img_array):
    result_img = img_array.copy()
    mask = (
        (-1, 0, 1),
        (-1, 0, 1),
        (-1, 0, 1),
    )
    for x, row in enumerate(img_array):
        for y, p in enumerate(row):
            new_pix = apply_mask(mask, img_array, coords=(x, y))
            result_img[x][y] = new_pix

    return result_img


def laplasian_method(img_array):
    result_img = img_array.copy()
    mask = (
        (0, 1, 0),
        (1, -4, 1),
        (0, 1, 0)
    )
    for x, row in enumerate(img_array):
        for y, p in enumerate(row):
            new_pix = apply_mask(mask, img_array, coords=(x, y))
            result_img[x][y] = new_pix

    return result_img


def log_filter(img_array):
    result_img = img_array.copy()
    # mask = [
    #     [0, 2, 2, 2, 0],
    #     [2, 1, -4, 1, 2],
    #     [2, -4, -20, -4, 2],
    #     [2, 1, -4, 1, 2],
    #     [0, 2, 2, 2, 0],
    # ]
    mask = [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0],
    ]
    # size = len(mask)
    # total = ((size * (size - 1)) / 2.0) * size
    for row in mask:
        total = sum(row)
    if abs(total - 1) > 0.01:
        k = 1.0 / total
        for x, row in enumerate(mask):
            for y, val in enumerate(row):
                mask[x][y] /= k

    for x, row in enumerate(img_array):
        for y, p in enumerate(row):
            new_pix = apply_mask(mask, img_array, coords=(x, y))
            result_img[x][y] = new_pix

    return result_img


def gaussian_diff(x, y, sig1, sig2):
    return ((exp(-1 * (pow(x, 2) + pow(y, 2)) / (2 * sig1)) / pow(sig1, 2)) -
            (exp(-1 * (pow(x, 2) + pow(y, 2)) / (2 * sig2)) / pow(sig2, 2))) / 2 * pi


def dog_filter(img_array, sig1, sig2):
    result_img = img_array.copy()

    size = 5
    mask = []

    total = 0
    for x in range(size):
        row = []
        for y in range(size):
            gd = gaussian_diff(y, x, sig1, sig2)
            row.append(gd)
            total += gd
        mask.append(row)

    if abs(total) > 0.01:
        k = 1.0 / total
        for row in mask:
            map(lambda a: a / k, row)

    for x, row in enumerate(img_array):
        for y, p in enumerate(row):
            new_pix = apply_mask(mask, img_array, coords=(x, y))
            result_img[x][y] = new_pix

    return result_img


def dog_filter(img_array, sig1, sig2):

    sig1 = int(sig1)
    sig2 = int(sig2)
    blur5 = cv.GaussianBlur(img_array, (sig1, sig1), 0)
    blur3 = cv.GaussianBlur(img_array, (sig2, sig2), 0)

    return blur5 - blur3


def sharpness_filter(img_array, diameter, sigma_i, sigma_s, a):
    result_img = img_array.copy()
    max_width, max_height = img_array.shape
    diameter_2 = diameter * 2

    for x, row in enumerate(img_array):
        for y, p in enumerate(row):
            f = img_array[x][y]
            if y > diameter_2 and x > diameter_2 and y < max_height - diameter_2 and x < max_width - diameter_2:
                fn = bilateral_filter(img_array, x, y, diameter, sigma_i, sigma_s)
            else:
                fn = f
            g = a * f - fn
            result_img[x][y] = g

    return result_img


def sharpness_filter(img_array):
    mask = np.array((
        (-1, 0, -1),
        (0, 5, 0),
        (-1, 0, -1)
    ))
    return cv.filter2D(img_array, -1, mask)
