import numpy as np


def reverse_even_rows(img_array):
    for x, row in enumerate(img_array):
        if x % 2:
            img_array[x] = row[::-1]


def my_adaptive_threshold(image_array, n, k):
    temp_image = image_array.copy()
    result_img = image_array.copy().flatten()
    n, k = float(n), float(k)

    MAX_VAL = np.uint8(255)
    MIN_VAl = np.uint8(0)

    m = 0.0
    # reverse_even_rows(temp_image)
    temp_image = temp_image.flatten()
    for i, p in enumerate(temp_image):
        if i < n:
            m = sum(temp_image[0:i + 1]) / float(len(temp_image[0:i + 1]))
        else:
            m = m + p / n - temp_image[int(i - n)] / n
        thresh = k * m
        result_img[i] = MIN_VAl if p <= thresh else MAX_VAL

    result_img = result_img.reshape(image_array.shape)
    # reverse_even_rows(result_img)
    return result_img
