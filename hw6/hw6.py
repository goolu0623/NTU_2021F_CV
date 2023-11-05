import numpy as np
from cv2 import cv2


def DownSample(image, DownSampleSize):
    DownSampleImage = np.zeros((DownSampleSize, DownSampleSize))
    # return DownSampleImage
    DownSamplelength = int(len(image) / DownSampleSize)
    for i in range(len(DownSampleImage)):
        for j in range(len(DownSampleImage[i])):
            DownSampleImage[i][j] = image[DownSamplelength * i][DownSamplelength * j]
    return DownSampleImage


def Binarize(image):
    BinarizeImage = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image)):
            BinarizeImage[i][j] = 0 if image[i][j] < 128 else 255
    return BinarizeImage


def Yokoi(image):
    YokoiImage = np.zeros(image.shape)
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] == 0:
                YokoiImage[i][j] = 0
                continue
            r, q = 0, 0
            # a1
            if j + 1 < len(image[i]) and image[i][j] == image[i][j + 1]:
                q += 1
            if i - 1 >= 0 and j + 1 < len(image[i]) and image[i][j] == image[i][j + 1] == image[i - 1][j] == image[i - 1][j + 1]:
                q -= 1
                r += 1
            # a2
            if i - 1 >= 0 and image[i][j] == image[i - 1][j]:
                q += 1
            if j - 1 >= 0 and i - 1 >= 0 and image[i][j] == image[i - 1][j] == image[i - 1][j - 1] == image[i][j - 1]:
                q -= 1
                r += 1
            # a3
            if j - 1 >= 0 and image[i][j] == image[i][j - 1]:
                q += 1
            if i + 1 < len(image) and j - 1 >= 0 and image[i][j] == image[i][j - 1] == image[i + 1][j] == image[i + 1][
                j - 1]:
                q -= 1
                r += 1
            # a4
            if i + 1 < len(image) and image[i][j] == image[i + 1][j]:
                q += 1
            if j + 1 < len(image[i]) and i + 1 < len(image) and image[i][j] == image[i + 1][j] == image[i][j + 1] == image[i + 1][j + 1]:
                q -= 1
                r += 1
            if r == 4:
                YokoiImage[i][j] = 5
            else:
                YokoiImage[i][j] = q
    return YokoiImage


def WriteResult(image):
    f = open("result.txt", 'w')
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] == 0:
                f.write("  ")
                print("  ", end="")
            else:
                f.write(str(image[i][j]).strip(".0"))
                print(str(image[i][j]).strip(".0"), end="")
                f.write(" ")
                print(" ", end="")
        f.write("\n")
        print("\n")
    f.close()


if __name__ == '__main__':
    lena = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    WriteResult(Yokoi(Binarize(DownSample(lena, 64))))
