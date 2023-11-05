from cv2 import cv2
import numpy as np
import math


def Roberts(image, threshold):
    RobertsImage = image.copy()
    imageBordered = cv2.copyMakeBorder(image, 0, 1, 0, 1, cv2.BORDER_REFLECT)  # Extend image borders
    for i in range(len(image)):
        for j in range(len(image[i])):
            RobertsImage[i][j] = 0 if (
                        math.sqrt(((int(imageBordered[i + 1][j + 1]) - int(imageBordered[i][j])) ** 2) + ((int(imageBordered[i + 1][j]) - int(imageBordered[i][j + 1])) ** 2)) >= threshold) else 255
    return RobertsImage


def Prewitt(image, threshold, p1mask, p2mask):
    PrewittImage = image.copy()
    imageBoarded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)  # Extend image borders
    for i in range(len(image)):
        for j in range(len(image[i])):
            p1sum, p2sum = 0, 0
            for r in range(len(p1mask)):
                for c in range(len(p1mask[r])):
                    p1sum += imageBoarded[i + r - 1][j + c - 1] * p1mask[r][c]
                    p2sum += imageBoarded[i + r - 1][j + c - 1] * p2mask[r][c]
            gradientMagnitude = math.sqrt(p1sum ** 2 + p2sum ** 2)
            PrewittImage[i][j] = 0 if gradientMagnitude >= threshold else 255
    return PrewittImage


def KirschCompass(image, threshold, mask):
    KirschCompassImage = image.copy()
    imageBoarded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)  # Extend image borders
    for i in range(len(image)):
        for j in range(len(image[i])):
            maxNum = -9999999999999
            for cnt in range(8):
                mask = rot33matrix(mask)
                KNum = 0
                for r in range(len(mask)):
                    for c in range(len(mask[r])):
                        KNum += int(imageBoarded[i + r - 1][j + c - 1]) * int(mask[r][c])
                maxNum = max(maxNum, KNum)
            KirschCompassImage[i][j] = 0 if maxNum >= threshold else 255
    return KirschCompassImage


def NevatiaAndBabu(image, threshold):
    mask1 = [[100, 100, 100, 100, 100], [100, 100, 100, 100, 100], [0, 0, 0, 0, 0], [-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100]]
    mask2 = [[100, 100, 100, 100, 100], [100, 100, 100, 78, -32], [100, 92, 0, -92, -100], [32, -78, -100, -100, -100], [-100, -100, -100, -100, -100]]
    mask3 = [[100, 100, 100, 32, -100], [100, 100, 92, -78, -100], [100, 100, 0, -100, -100], [100, 78, -92, -100, -100], [100, -32, -100, -100, -100]]
    mask4 = [[-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100]]
    mask5 = [[-100, 32, 100, 100, 100], [-100, -78, 92, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, -92, 78, 100], [-100, -100, -100, -32, 100]]
    mask6 = [[100, 100, 100, 100, 100], [-32, 78, 100, 100, 100], [-100, -92, 0, 92, 100], [-100, -100, -100, -78, 32], [-100, -100, -100, -100, -100]]
    NevatiaImage = image.copy()
    imageBoarded = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REFLECT)  # Extend image borders
    for i in range(len(image)):
        for j in range(len(image[i])):
            KNum1, KNum2, KNum3, KNum4, KNum5, KNum6 = 0, 0, 0, 0, 0, 0
            for r in range(len(mask1)):
                for c in range(len(mask1[r])):
                    KNum1 += int(imageBoarded[i + r - 2][j + c - 2]) * int(mask1[r][c])
                    KNum2 += int(imageBoarded[i + r - 2][j + c - 2]) * int(mask2[r][c])
                    KNum3 += int(imageBoarded[i + r - 2][j + c - 2]) * int(mask3[r][c])
                    KNum4 += int(imageBoarded[i + r - 2][j + c - 2]) * int(mask4[r][c])
                    KNum5 += int(imageBoarded[i + r - 2][j + c - 2]) * int(mask5[r][c])
                    KNum6 += int(imageBoarded[i + r - 2][j + c - 2]) * int(mask6[r][c])
            maxNum = max(KNum1, KNum2, KNum3, KNum4, KNum5, KNum6)
            NevatiaImage[i][j] = 0 if maxNum >= threshold else 255
    return NevatiaImage


def rot33matrix(matrix):
    temp = matrix[0][0]
    matrix[0][0] = matrix[0][1]
    matrix[0][1] = matrix[0][2]
    matrix[0][2] = matrix[1][2]
    matrix[1][2] = matrix[2][2]
    matrix[2][2] = matrix[2][1]
    matrix[2][1] = matrix[2][0]
    matrix[2][0] = matrix[1][0]
    matrix[1][0] = temp
    return matrix


def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

    # Robert
    RobertsImage = Roberts(lena, 30)

    # Prewitt
    p1mask = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    p2mask = np.rot90(p1mask)
    PrewittImage = Prewitt(lena, 24, p1mask, p2mask)

    # Sobel
    p1mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    p2mask = np.rot90(p1mask)
    SobelImage = Prewitt(lena, 38, p1mask, p2mask)

    # Frei and Chen
    p1mask = np.array([[-1, -math.sqrt(2), -1], [0, 0, 0], [1, math.sqrt(2), 1]])
    p2mask = np.rot90(p1mask)
    FreiAndChenImage = Prewitt(lena, 30, p1mask, p2mask)

    # Kirsch compass
    mask = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    KirschCompassImage = KirschCompass(lena, 135, mask)

    # Robinson
    mask = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    RobinsonImage = KirschCompass(lena, 43, mask)

    # Nevatia-Babu
    NevatiaAndBabuImage = NevatiaAndBabu(lena, 12500)

    # show all image
    cv2.imshow("Roberts", RobertsImage)
    cv2.imshow("Pre", PrewittImage)
    cv2.imshow("Sobel", SobelImage)
    cv2.imshow("Frei and Chen", FreiAndChenImage)
    cv2.imshow("Kirsch Compass", KirschCompassImage)
    cv2.imshow("Robinson", RobinsonImage)
    cv2.imshow("Nevatia-Babu", NevatiaAndBabuImage)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # save all image
    cv2.imwrite("Roberts.bmp", RobertsImage)
    cv2.imwrite("Pre.bmp", PrewittImage)
    cv2.imwrite("Sobel.bmp", SobelImage)
    cv2.imwrite("Frei and Chen.bmp", FreiAndChenImage)
    cv2.imwrite("Kirsch Compass.bmp", KirschCompassImage)
    cv2.imwrite("Robinson.bmp", RobinsonImage)
    cv2.imwrite("Nevatia-Babu.bmp", NevatiaAndBabuImage)


def test():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    NevatiaAndBabuImage = NevatiaAndBabu(lena, 12500)
    cv2.imshow("Nevatia-Babu", NevatiaAndBabuImage)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test()
    main()
