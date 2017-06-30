from collections import Counter


def pixels_of(img_array):
    for row in img_array:
        for pixel in row:
            yield pixel


def count_pixels(img_array):
    total_count = Counter()
    for row in img_array:
        total_count = total_count + Counter(row)
    return total_count
