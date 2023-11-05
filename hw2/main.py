import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt


def binary():
    for r in range(rows):
        for c in range(columns):
            gray = lenaCopy[r, c]
            if gray < 128:
                lenaCopy[r, c] = 0
            else:
                lenaCopy[r, c] = 255


# <-- Display image -->
# cv2.imshow("binary", lenaCopy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# <-- Save image -->
# cv2.imwrite("answer1.bmp",lenaCopy)


def drawHistogram():
    histData = []
    for r in range(rows):
        for c in range(columns):
            gray = 0.299 * lena[r, c, 2] + 0.587 * lena[r, c, 1] + 0.114 * lena[r, c, 0]
            if gray > 255:
                gray = 255
            histData.append(int(gray))

    n, bins, patches = plt.hist(x=histData, bins=256, histtype='bar', density=False)
    plt.title(r'Histogram')
    plt.savefig("answer2.png")
    plt.show()


def find(data, i):
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]


def union(data, i, j):
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        data[pj] = pi


def connected(data, i, j):
    return find(data, i) == find(data, j)


def findCentroid(labelList, connectLabel):
    centerCal = np.zeros([len(labelList), 2], dtype=int)
    recordBox = np.array([[rows, columns, 0, 0], [rows, columns, 0, 0], [rows, columns, 0, 0], [rows, columns, 0, 0],
                          [rows, columns, 0, 0]])
    for number in range(len(labelList)):
        for r in range(rows):
            for c in range(columns):
                if connectLabel[r, c] == list(labelList.keys())[number]:
                    centerCal[number, 0] += r
                    centerCal[number, 1] += c

                    if recordBox[number][0] > r:
                        recordBox[number][0] = r
                    if recordBox[number][2] < r:
                        recordBox[number][2] = r
                    if recordBox[number][1] > c:
                        recordBox[number][1] = c
                    if recordBox[number][3] < c:
                        recordBox[number][3] = c
    lena2 = cv2.cvtColor(lenaCopy, cv2.COLOR_GRAY2RGB)
    for number in range(len(centerCal)):
        # Average the total axis to find centroid
        centerCal[number, 0] = centerCal[number, 0] / list(labelList.values())[number]
        centerCal[number, 1] = centerCal[number, 1] / list(labelList.values())[number]
        # Draw circle on centroid
        cv2.circle(lena2, (centerCal[number, 1], centerCal[number, 0]), 7, (28, 46, 184), -1)
        # Draw bounding box
        cv2.rectangle(lena2, (recordBox[number][1], recordBox[number][0]), (recordBox[number][3], recordBox[number][2]),
                      (247, 196, 27), 2)



    # print(centerCal)
    cv2.imshow("Circle", lena2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("answer3.bmp", lena2)


# Two-pass
def connectedComp():
    connectLabel = np.zeros([512, 512], dtype=int)
    labels = 0
    data = [i for i in range(3000)]

    for r in range(rows):
        for c in range(columns):
            # White pixel
            if lenaCopy[r, c] == 255:
                # None-border pixels
                if c != 0 and r != 0:
                    if lenaCopy[r, c - 1] == 255 and lenaCopy[r - 1, c] == 255:
                        connectLabel[r, c] = min(connectLabel[r, c - 1], connectLabel[r - 1, c])
                        union(data, connectLabel[r, c], max(connectLabel[r, c - 1], connectLabel[r - 1, c]))
                    elif lenaCopy[r, c - 1] == 255:
                        connectLabel[r, c] = connectLabel[r, c - 1]
                    elif lenaCopy[r - 1, c] == 255:
                        connectLabel[r, c] = connectLabel[r - 1, c]
                    else:
                        labels += 1
                        connectLabel[r, c] = labels

                # First pixel
                elif r == 0 and c == 0:
                    if lenaCopy[0, 0] == 255:
                        labels += 1
                        connectLabel[r, c] = labels

                # Left-side pixels
                elif c == 0:
                    if lenaCopy[r - 1, c] == 255:
                        connectLabel[r, c] = connectLabel[r - 1, c]
                    else:
                        labels += 1
                        connectLabel[r, c] = labels

                # Top-side pixels
                elif r == 0:
                    if lenaCopy[r, c - 1] == 255:
                        connectLabel[r, c] = connectLabel[r, c - 1]
                    else:
                        labels += 1
                        connectLabel[r, c] = labels

    # Union-Find
    labelList = {}
    for r in range(rows):
        for c in range(columns):
            if connectLabel[r, c] != 0:
                connectLabel[r, c] = find(data, connectLabel[r, c])
                if connectLabel[r, c] not in labelList:
                    labelList[connectLabel[r, c]] = 1
                else:
                    labelList[connectLabel[r, c]] += 1

    # Threshold = 500
    for item in list(labelList.keys()):
        if labelList[item] < 500:
            del labelList[item]

    # print(labelList)
    findCentroid(labelList, connectLabel)


if __name__ == '__main__':


    lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
    rows, columns = lena.shape[:2]
    lenaCopy = np.zeros(lena.shape, np.uint8)
    lenaCopy = lena.copy()
    binary()
    # drawHistogram()
    connectedComp()
