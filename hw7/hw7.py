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
            if i - 1 >= 0 and j + 1 < len(image[i]) and image[i][j] == image[i][j + 1] == image[i - 1][j] == \
                    image[i - 1][j + 1]:
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
            if j + 1 < len(image[i]) and i + 1 < len(image) and image[i][j] == image[i + 1][j] == image[i][j + 1] == \
                    image[i + 1][j + 1]:
                q -= 1
                r += 1
            if r == 4:
                YokoiImage[i][j] = 5
            else:
                YokoiImage[i][j] = q
    return YokoiImage


def WriteResult(image):
    # f = open("result.txt", 'w')
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] == 0:
                # f.write("  ")
                print(0, end="")
                # print(" ", end="")
                print(" ", end="")
            else:
                # f.write(str(image[i][j]).strip(".0"))
                print(str(image[i][j]).strip(".0"), end="")
                # f.write(" ")
                print(" ", end="")
        # f.write("\n")
        print("\n")
    # f.close()


def PairRelationship(image):
    PairImage = np.zeros(image.shape)  # 這裡拿q=1 , p=2
    list = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] == 0:
                continue
            PairImage[i][j] = 1
            if image[i][j] == 1:
                for position in list:
                    r, c = position
                    if 0 <= i + r < len(image) and 0 <= j + c < len(image[i]) and image[i][j] == image[i + r][j + c]:
                        PairImage[i][j] = 2
                        break
    return PairImage


def thinning(image, thinningImage):
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] <= 1:  # q=1, p=2 我只處理p
                continue
            r, q = 0, 0
            # a1
            if j + 1 < len(image[i]) and thinningImage[i][j] == thinningImage[i][j + 1]:
                q += 1
            if i - 1 >= 0 and j + 1 < len(image[i]) and thinningImage[i][j] == thinningImage[i][j + 1] == \
                    thinningImage[i - 1][j] == \
                    thinningImage[i - 1][j + 1]:
                q -= 1
                r += 1
            # a2
            if i - 1 >= 0 and thinningImage[i][j] == thinningImage[i - 1][j]:
                q += 1
            if j - 1 >= 0 and i - 1 >= 0 and thinningImage[i][j] == thinningImage[i - 1][j] == thinningImage[i - 1][
                j - 1] == thinningImage[i][
                j - 1]:
                q -= 1
                r += 1
            # a3
            if j - 1 >= 0 and thinningImage[i][j] == thinningImage[i][j - 1]:
                q += 1
            if i + 1 < len(image) and j - 1 >= 0 and thinningImage[i][j] == thinningImage[i][j - 1] == \
                    thinningImage[i + 1][j] == \
                    thinningImage[i + 1][j - 1]:
                q -= 1
                r += 1
            # a4
            if i + 1 < len(image) and thinningImage[i][j] == thinningImage[i + 1][j]:
                q += 1
            if j + 1 < len(image[i]) and i + 1 < len(image) and thinningImage[i][j] == thinningImage[i + 1][j] == \
                    thinningImage[i][j + 1] == \
                    thinningImage[i + 1][j + 1]:
                q -= 1
                r += 1
            if q == 1:
                thinningImage[i][j] = 0
                image[i][j] = 0
    return thinningImage


def ScaledUp(image, size):
    ScaledUpImage = np.zeros((size, size))
    ScaledUpParameter = int(size / len(image))
    for i in range(len(ScaledUpImage)):
        for j in range(len(ScaledUpImage[i])):
            ScaledUpImage[i][j] = image[int(i / ScaledUpParameter)][int(j / ScaledUpParameter)]
    return ScaledUpImage


if __name__ == '__main__':

    lena = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    # 先down sample成64*64 再binarize
    finalImage = Binarize(DownSample(lena, 64))
    for i in range(7):
        # 做Yokoi標出每個點的關係
        shrinkImage = Yokoi(finalImage)
        # pair relation
        shrinkImage = PairRelationship(shrinkImage)
        # thinning
        finalImage = thinning(shrinkImage, finalImage)
    finalImage = ScaledUp(finalImage, len(lena))

    cv2.imwrite('thinning_image.bmp', finalImage)
    cv2.imshow('thinnig', finalImage)
    cv2.waitKey()
    cv2.destroyAllWindows()
