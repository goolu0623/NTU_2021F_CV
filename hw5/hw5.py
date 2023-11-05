import numpy as np
from cv2 import cv2


def octa_kernel():
    return [[-2, 1], [-2, 0], [-2, -1],
            [-1, 2], [-1, 1], [-1, 0], [-1, -1], [-1, -2],
            [0, 2], [0, 1], [0, 0], [0, -1], [0, -2],
            [1, 2], [1, 1], [1, 0], [1, -1], [1, -2],
            [2, 1], [2, 0], [2, -1]]


def dilation(image, kernel):
    dilimage = np.zeros(image.shape, np.uint8)
    m, n = image.shape
    for i in range(m):
        for j in range(n):
            kernel_max = 0
            kernel_max2=0
            for position in kernel:
                p, q = position
                if 0 <= (i + p) <= (m - 1) and 0 <= (j + q) <= (n - 1):
                    kernel_max = max(image[i + p][j + q], kernel_max)
            dilimage[i][j] = kernel_max
    return dilimage


def erosion(image, kernel):
    lena_erosion = np.zeros(image.shape, np.uint8)
    rows, columns = lena_erosion.shape
    for i in range(rows):
        for j in range(columns):
            kernel_min = 255
            for position in kernel:
                p, q = position
                if 0 <= (i + p) < 512 and 0 <= (j + q) < 512:
                    kernel_min = min(image[i + p][j + q], kernel_min)

            lena_erosion[i][j] = kernel_min
    return lena_erosion


def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)


def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)


def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    kernel = octa_kernel()
    cv2.imwrite("erosion.bmp", erosion(lena, kernel))
    cv2.imwrite("dilation.bmp", dilation(lena, kernel))
    cv2.imwrite("opening.bmp", opening(lena, kernel))
    cv2.imwrite("closing.bmp", closing(lena, kernel))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
