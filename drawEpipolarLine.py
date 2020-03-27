import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def main():
    imageOne = mpimg.imread('/Users/vikrampasupathy/Downloads/leftright/left.jpg')
    plt.imshow(imageOne)
    pointsFromLeft = plt.ginput(10)
    imageTwo = mpimg.imread('/Users/vikrampasupathy/Downloads/leftright/right.jpg')
    plt.imshow(imageTwo)
    throwawaypoint = plt.ginput(1)
    pointsFromRight = plt.ginput(10)
    fundmatrix = cv2.findFundamentalMat(np.float32(pointsFromLeft), np.float32(pointsFromRight), cv2.FM_LMEDS)
    hardcodedfundMatrix = [[ 4.71943093e-06,  1.86904908e-04, -3.02366130e-05],
       [-2.42578372e-04,  1.04289172e-04,  6.82730006e-02],
       [-5.45289813e-03, -7.54837686e-02,  1.00000000e+00]]
    
    drawEpipolarLine(imageTwo, hardcodedfundMatrix, pointsFromLeft[0])
 
def drawEpipolarLine(imageTwo, hardcodedfundMatrix, point):
    point = [point[0], point[1], 1]
    np.matrix(hardcodedfundMatrix)
    np.matrix(point)
    line = np.dot(hardcodedfundMatrix, point)
    plt.imshow(imageTwo)
    a, b, c = line.ravel()
    x = np.array([0,400])
    y = -(x*a+c)/b
    plt.plot(x,y)

    
if __name__ == "__main__":
    main()