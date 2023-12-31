import cv2
import numpy as np


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


img = cv2.imread('data/ntu_dark_extracted/Drink/Drink_3_1/img_00001.jpg')
gammaImg = gammaCorrection(img, 2.2)

cv2.imwrite('tools/data/skeleton/GID_eg.jpg',gammaImg)
# cv2.imshow('Original image', img)
# cv2.imshow('Gamma corrected image', gammaImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()