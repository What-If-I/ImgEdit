def average_blur_filter(image_array, n):
    assert n > 1

    result_img = image_array.copy()
    threshold = n - 1

    for x, row in enumerate(image_array):
        for y, p in enumerate(row):
            mask_width = x - threshold, x + threshold + 1
            mask_height = y - threshold, y + threshold + 1  # Get rid of that +1
            mask_pixels = []
            for mask_x, mask_y in zip(xrange(*mask_width), xrange(*mask_height)):
                try:
                    pix = result_img[mask_x][mask_y]
                except IndexError:
                    pass  # skip non-existing indexes
                else:
                    mask_pixels.append(pix)

            avg = sum(mask_pixels) / len(mask_pixels)
            result_img[x][y] = avg

    return result_img
