import rasterio
import matplotlib.pyplot as plt
import numpy as np
from commom import *
from utils import *
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math


def getMap(fileName):
    with rasterio.open(fileName, 'r') as ds:
        img = ds.read()[0]  # read all raster values
    return np.matrix(img)

def getSubregion(map, x0, y0, x1, y1):
    return map[int(y0):int(y1), int(x0):int(x1)]

def applyNoise(img, std=1):
    h, w = img.shape
    noisyImg = np.random.randn(h, w)
    noisyImg = noisyImg*std + img
    return noisyImg

def drawRectangle(img, x0, y0, x1, y1, color=(255, 255, 255)):     # TODO finish this function
    height, width = img.shape
    if width >= height: max = width
    else:  max = height
    thickness = int(max*0.001)
    img = cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    return img

def showSliceOfImg(img, x0=0.5, show=True):
    h, w = img.shape
    slice = img[:, int(x0*w)]
    if show:
        plt.plot(slice)
        plt.grid()
        plt.show()
    return slice

def insidePolygon(p, x0, y0, x1, y1):   # TODO Not being used?
    xp, yp = p
    point = Point(xp, yp)
    polygon = Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    return polygon.contains(point)

def rotateMap(map, deg):
    height, width = map.shape
    cr = (width/2,height/2)     # define center of rotation
    M = cv2.getRotationMatrix2D(cr,deg,1)    # 'deg' in anticlockwise direction
    return cv2.warpAffine(map,M,(width,height))  # apply warpAffine() method to perform image rotation

def plotSideBySide(data1, data2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(data1)
    ax2.plot(data2)
    plt.grid()
    plt.plot()

def rotate2DRectangle(x0, x1, y0, y1, cx, cy, deg):
    points = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
    return rotateAroundCenter(points, cx, cy, deg)
     
def drawPolygon(img, pts, color=(255, 255, 255)):
    img = np.int32(img)
    height, width = img.shape
    if width >= height: max = width
    else:  max = height
    thickness = int(max*0.001)
    img = cv2.polylines(img, np.int32([pts]), 1, color, thickness)
    return np.uint8(img)

def rotateAroundCenter(pts, cx, cy, ang):
    radians = np.radians(ang)
    cos, sin = math.cos(radians), math.sin(radians)
    rotPts = []
    for p in pts:
        x, y = p
        rx = cx + cos * (x - cx) + sin * (y - cy)
        ry = cy + -sin * (x - cx) + cos * (y - cy)
        rotPts.append([rx, ry])
    return np.array(rotPts)

def countInliers(matches, pts):
    inliers = 0
    for match in matches:
        point = Point(match.pt[0], match.pt[1])
        polygon = Polygon(pts)
        if polygon.contains(point): inliers += 1
    return inliers


def calcNumberLevels(imgShape, scaleFactor, desiredFinalArea):
    '''Used to calculate pyramid parameters in
    ORB Detector.'''
    A0 = imgShape[0]*imgShape[1]
    return int(math.log(A0/desiredFinalArea, scaleFactor))
