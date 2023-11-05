import numpy as np
import cv2.cv2 as cv2


def binary(image):
    lena_binary = np.zeros(image.shape, np.uint8)
    rows, columns = lena_binary.shape
    for i in range(rows):
        for j in range(columns):
            if image[i][j] >= 128:
                lena_binary[i][j] = 255
            else:
                lena_binary[i][j] = 0
    return lena_binary


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
            if image[i][j] > 0:
                for position in kernel:
                    p, q = position
                    if 0 <= (i + p) <= (m - 1) and 0 <= (j + q) <= (n - 1):
                        dilimage[i + p][j + q] = 255
    return dilimage


def erosion(image, kernel):
    lena_erosion = np.zeros(image.shape, np.uint8)
    rows, columns = lena_erosion.shape
    for i in range(rows):
        for j in range(columns):
            flag = True
            for position in kernel:
                p, q = position
                if 0 <= (i + p) < 512 and 0 <= (j + q) < 512 and image[i + p, j + q] == 0:
                    flag = False
                    break
            if flag:
                lena_erosion[i][j] = 255
    return lena_erosion


# def opening(image, kernel):
#     opimage = np.zeros(image.shape, np.uint8)
#     m, n = image.shape
#     for i in range(m):
#         for j in range(n):
#             if image[i][j] > 0:
#                 flag = False
#                 for position in kernel:
#                     p, q = position
#                     if 0 <= (i + p) <= (m - 1) and 0 <= (j + q) <= (n - 1) and image[i + p][
#                         j + q] == 0:
#                         flag = True
#                 if flag == False:
#                     for position in kernel:
#                         p, q = position
#                         if 0 <= (i + p) <= (m - 1) and 0 <= (j + q) <= (n - 1):
#                             opimage[i + p][j + q] = 255
#     return opimage


def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)


def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)


# def closing(image, kernel):
#     closeimage = np.zeros(image.shape, np.uint8)
#     m, n = image.shape
#     for i in range(m):
#         for j in range(n):
#             closeimage[i][j] = 255
#     for i in range(m):
#         for j in range(n):
#             if image[i][j] == 0:
#                 flag = False
#                 for position in kernel:
#                     p, q = position
#                     if 0 <= (i + p) <= (m - 1) and 0 <= (j + q) <= (n - 1) and image[i + p][j + q] > 0:
#                         flag = True
#                 if flag == False:
#                     for position in kernel:
#                         p, q = position
#                         if 0 <= (i + p) <= (m - 1) and 0 <= (j + q) <= (n - 1):
#                             closeimage[i + p][j + q] = 0
#     return closeimage


def hit_and_miss(image, jker, kker):
    image_complement = 255 - image
    eroj = erosion(image, jker)
    erok = erosion(image_complement, kker)
    ham_image = np.zeros(image.shape, np.uint8)
    for i in range(len(image)):
        for j in range(len(image[i])):
            ham_image[i][j] = eroj[i][j] & erok[i][j]
    return ham_image


def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    lena_binary = binary(lena)
    kernel = octa_kernel()
    jker = [[0, -1], [0, 0], [1, 0]]
    kker = [[-1, 0], [-1, 1], [0, 1]]
    cv2.imwrite("erosion.bmp", erosion(lena_binary, kernel))
    cv2.imwrite("dilation.bmp", dilation(lena_binary, kernel))
    cv2.imwrite("opening.bmp", opening(lena_binary, kernel))
    cv2.imwrite("closing.bmp", closing(lena_binary, kernel))
    cv2.imwrite("hit_and_miss.bmp", hit_and_miss(lena_binary, jker, kker))
    # cv2.imshow("hit_and_miss.bmp", hit_and_miss(lena_binary, jker, kker))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
