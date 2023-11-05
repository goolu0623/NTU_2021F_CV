from cv2 import cv2


def Laplcian(image, mask, threshold):
    imageBoarded = cv2.copyMakeBorder(image, int(len(mask) / 2), int(len(mask) / 2), int(len(mask[0]) / 2), int(len(mask[0]) / 2), cv2.BORDER_REFLECT)  # Extend image borders
    MaskPixel = image.copy()
    for i in range(len(image)):
        for j in range(len(image[i])):
            sum = 0
            for m in range(len(mask)):
                for n in range(len(mask[m])):
                    sum += mask[m][n] * int(imageBoarded[i + m][j + n])
            if sum >= threshold:
                MaskPixel[i][j] = 2
            elif sum <= -threshold:
                MaskPixel[i][j] = 0
            else:
                MaskPixel[i][j] = 1
    BorderExtendMaskPixel = cv2.copyMakeBorder(MaskPixel, 1, 1, 1, 1, cv2.BORDER_REFLECT)  # Extend image borders
    ImageReturn = image.copy()
    crossmask = [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    for i in range(len(image)):
        for j in range(len(image[i])):
            ImageReturn[i][j] = 255
            if BorderExtendMaskPixel[i + 1][j + 1] != 2:
                continue
            for position in crossmask:
                a, b = position
                if BorderExtendMaskPixel[i + 1 + a][j + 1 + b] == 0:
                    ImageReturn[i][j] = 0
                    break
    return ImageReturn


def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    mask1 = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    mask2 = [[0.333, 0.333, 0.333], [0.333, -2.666, 0.333], [0.333, 0.333, 0.333]]
    mask3 = [[0.666, -0.333, 0.666], [-0.333, -1.333, -0.333], [0.666, -0.333, 0.666]]
    mask4 = [[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
             [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
             [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
             [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
             [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
             [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
             [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
             [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
             [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
             [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
             [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]]
    mask5 = [[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
             [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
             [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
             [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
             [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
             [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
             [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
             [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
             [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
             [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
             [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]]
    LaplaceM1 = Laplcian(lena, mask1, 15)
    LaplaceM2 = Laplcian(lena, mask2, 15)
    MiniLaplace = Laplcian(lena, mask3, 20)
    LaplaceGau = Laplcian(lena, mask4, 3000)
    DiffGau = Laplcian(lena, mask5, 1)

    cv2.imshow("1", LaplaceM1)
    cv2.imshow("2", LaplaceM2)
    cv2.imshow("3", MiniLaplace)
    cv2.imshow("4", LaplaceGau)
    cv2.imshow("5", DiffGau)

    cv2.imwrite("Laplace_Mask1.bmp", LaplaceM1)
    cv2.imwrite("Laplace_Mask2.bmp", LaplaceM2)
    cv2.imwrite("miniLaplace.bmp", MiniLaplace)
    cv2.imwrite("Laplace_Gaussian.bmp", LaplaceGau)
    cv2.imwrite("Difference_Gaussian.bmp", DiffGau)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
