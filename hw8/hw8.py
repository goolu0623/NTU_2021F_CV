import cv2
import hw5
import random as rand
import math
import statistics


def snrCalculate(sImage, nImage):
    sImage = sImage.astype(int)
    nImage = nImage.astype(int)
    meanS = sImage.mean()
    meanN = (nImage - sImage).mean()
    vs = ((sImage - meanS) ** 2).mean()
    vn = ((nImage - sImage - meanN) ** 2).mean()
    return 20 * math.log10(math.sqrt(vs) / math.sqrt(vn))


def GaussianNoise(image, amplitude):
    GaussianImage = image.copy()
    for i in range(len(image)):
        for j in range(len(image[i])):
            noiseOutput = int(GaussianImage[i, j]) + (amplitude * rand.gauss(0, 1))
            if noiseOutput > 255:
                noiseOutput = 255
            if noiseOutput < 0:
                noiseOutput = 0
            GaussianImage[i, j] = noiseOutput

    print("Gaussian", amplitude, " SNR = ", snrCalculate(image, GaussianImage))
    return GaussianImage


def SaltandPepper(image, threshold):
    SPImage = image.copy()
    for i in range(len(image)):
        for j in range(len(image[i])):
            randomNumber = rand.uniform(0, 1)
            if randomNumber < threshold:
                SPImage[i, j] = 0
            elif randomNumber > (1 - threshold):
                SPImage[i, j] = 255

    print("S&P", threshold, " SNR = ", snrCalculate(image, SPImage))
    return SPImage


def boxFilter(FilterWidth, FilterHeight, Image):
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    FilterImage = Image.copy()
    Vcenter = int(FilterHeight / 2)
    Hcenter = int(FilterWidth / 2)
    for i in range(len(Image)):
        for j in range(len(Image[i])):
            sum = 0
            for r in range(-Vcenter, Vcenter + 1):
                for c in range(-Hcenter, Hcenter + 1):
                    if 0 <= i + r < len(Image) and 0 <= j + c < len(Image[i]):
                        sum += Image[i + r, j + c]
            FilterImage[i, j] = sum / (FilterWidth * FilterHeight)
    print("box Filter", FilterWidth, "x", FilterHeight, " SNR = ", snrCalculate(lena, FilterImage))
    return FilterImage


def medianFilter(FilterWidth, FilterHeight, Image):
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    FilterImage = Image.copy()
    Vcenter = int(FilterHeight / 2)
    Hcenter = int(FilterWidth / 2)
    for i in range(len(Image)):
        for j in range(len(Image[i])):
            pixels = []
            for r in range(-Vcenter, Vcenter + 1):
                for c in range(-Hcenter, Hcenter + 1):
                    if 0 <= i + r < len(Image) and 0 <= j + c < len(Image[i]):
                        pixels.append(int(Image[i + r, j + c]))
            FilterImage[i, j] = statistics.median(pixels)
    print("median Filter", FilterWidth, "x", FilterHeight, " SNR = ", snrCalculate(lena, FilterImage))
    return FilterImage


def main():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    kernel = hw5.octa_kernel()

    gaussianNoise10 = GaussianNoise(lena, 10)
    gaussianNoise30 = GaussianNoise(lena, 30)
    saltAndPepperNoise005 = SaltandPepper(lena, 0.05)
    saltAndPepperNoise010 = SaltandPepper(lena, 0.1)

    # Gaussian Noise
    cv2.imwrite("GaussianNoise10.bmp", gaussianNoise10)
    cv2.imwrite("GaussianNoise30.bmp", gaussianNoise30)

    # Salt and Pepper Noise
    cv2.imwrite("SaltAndPepperNoise005.bmp", saltAndPepperNoise005)
    cv2.imwrite("SaltAndPepperNoise010.bmp", saltAndPepperNoise010)

    # Box Filter 3x3
    cv2.imwrite("Gaussian10box33.bmp", boxFilter(3, 3, gaussianNoise10))
    cv2.imwrite("Gaussian30box33.bmp", boxFilter(3, 3, gaussianNoise30))
    cv2.imwrite("Salt005box33.bmp", boxFilter(3, 3, saltAndPepperNoise005))
    cv2.imwrite("Salt010box33.bmp", boxFilter(3, 3, saltAndPepperNoise010))

    # Box Filter 5x5
    cv2.imwrite("Gaussian10box55.bmp", boxFilter(5, 5, gaussianNoise10))
    cv2.imwrite("Gaussian30box55.bmp", boxFilter(5, 5, gaussianNoise30))
    cv2.imwrite("Salt005box55.bmp", boxFilter(5, 5, saltAndPepperNoise005))
    cv2.imwrite("Salt010box55.bmp", boxFilter(5, 5, saltAndPepperNoise010))

    # Median Filter 3x3
    cv2.imwrite("Gaussian10median33.bmp", medianFilter(3, 3, gaussianNoise10))
    cv2.imwrite("Gaussian30median33.bmp", medianFilter(3, 3, gaussianNoise30))
    cv2.imwrite("Salt005median33.bmp", medianFilter(3, 3, saltAndPepperNoise005))
    cv2.imwrite("Salt010median33.bmp", medianFilter(3, 3, saltAndPepperNoise010))

    # Median Filter 5x5
    cv2.imwrite("Gaussian10median55.bmp", medianFilter(5, 5, gaussianNoise10))
    cv2.imwrite("Gaussian30median55.bmp", medianFilter(5, 5, gaussianNoise30))
    cv2.imwrite("Salt005median55.bmp", medianFilter(5, 5, saltAndPepperNoise005))
    cv2.imwrite("Salt010median55.bmp", medianFilter(5, 5, saltAndPepperNoise010))

    temp = hw5.closing(hw5.opening(gaussianNoise10, kernel), kernel)
    print("open-close SNR = ", snrCalculate(lena, temp))
    cv2.imwrite("Gaussian10_OpenClose.bmp", temp)
    temp = hw5.opening(hw5.closing(gaussianNoise10, kernel), kernel)
    print("close-open SNR = ", snrCalculate(lena, temp))
    cv2.imwrite("Gaussian10_CloseOpen.bmp", temp)

    temp = hw5.closing(hw5.opening(gaussianNoise30, kernel), kernel)
    print("open-close SNR = ", snrCalculate(lena, temp))
    cv2.imwrite("Gaussian30_OpenClose.bmp", temp)
    temp = hw5.opening(hw5.closing(gaussianNoise30, kernel), kernel)
    print("close-open SNR = ", snrCalculate(lena, temp))
    cv2.imwrite("Gaussian30_CloseOpen.bmp", temp)

    temp = hw5.closing(hw5.opening(saltAndPepperNoise005, kernel), kernel)
    print("open-close SNR = ", snrCalculate(lena, temp))
    cv2.imwrite("Salt005_OpenClose.bmp", temp)
    temp = hw5.opening(hw5.closing(saltAndPepperNoise005, kernel), kernel)
    print("close-open SNR = ", snrCalculate(lena, temp))
    cv2.imwrite("Salt005_CloseOpen.bmp", temp)

    temp = hw5.closing(hw5.opening(saltAndPepperNoise010, kernel), kernel)
    print("open-close SNR = ", snrCalculate(lena, temp))
    cv2.imwrite("Salt010_OpenClose.bmp", temp)
    temp = hw5.opening(hw5.closing(saltAndPepperNoise010, kernel), kernel)
    print("close-open SNR = ", snrCalculate(lena, temp))
    cv2.imwrite("Salt010_CloseOpen.bmp", temp)


def test():
    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    gaussianNoise10 = GaussianNoise(lena, 30)
    cv2.imshow("123", gaussianNoise10)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    # test()
