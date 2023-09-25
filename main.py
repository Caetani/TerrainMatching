import rasterio
import matplotlib.pyplot as plt
import numpy as np
from commom import *
from utils import *
import cv2

files = ['map_data/1_2_dem.tif',
         'merged_map.tif'
]
f = files[0]

imgOriginal = getMap(f)

for i in range(10):
    img = np.uint8(imgOriginal)

    # Get sub-region
    height, width = img.shape

    div = 4
    showHeight, showWidhth = int(height/div), int(width/div)

    x0, x1 = 0.5, 0.7
    y0, y1 = 0.3, 0.4

    x0, x1 = int(x0*width), int(x1*width)
    y0, y1 = int(y0*height), int(y1*height)

    rotAngle = 10

    imgRot = rotateMap(img, rotAngle)
    sub = getSubregion(imgRot, x0, y0, x1, y1)
    sub = np.uint8(applyNoise(sub))

    sigma = 2
    windowSize = 3*sigma#int(5.1*sigma)
    if windowSize % 2 == 0: windowSize = windowSize + 1
    print(windowSize)
    img = cv2.GaussianBlur(img, (windowSize, windowSize), sigmaX=sigma, sigmaY=sigma)
    sub = cv2.GaussianBlur(sub, (windowSize, windowSize), sigmaX=sigma, sigmaY=sigma)

    #img = cv2.Canny(img, threshold1=t1, threshold2=t2)
    #sub = cv2.Canny(sub, threshold1=t1, threshold2=t2)

    img = 10*cv2.Laplacian(img, ddepth=cv2.CV_8U, ksize=3)
    sub = 10*cv2.Laplacian(sub, ddepth=cv2.CV_8U, ksize=3)

    #showImage(cv2.resize(imgCliff, dsize=(showWidhth, showHeight)))
    numFeatures = 1_000

    desiredFinalArea = 2_000
    scaleFactor = 1.2
    nLevelsImg = calcNumberLevels(img.shape, scaleFactor=scaleFactor, desiredFinalArea=desiredFinalArea)
    nLevelsSub = calcNumberLevels(sub.shape, scaleFactor=scaleFactor, desiredFinalArea=desiredFinalArea)

    print(nLevelsImg, nLevelsSub)

    imgKeypoints, imgDescriptor = orbDetectorAndDescriptor(img, numFeatures, scaleFactor=scaleFactor, nlevels=nLevelsImg, showImage=False)
    subKeypoints, subDescriptor = orbDetectorAndDescriptor(sub, numFeatures, scaleFactor=scaleFactor, nlevels=nLevelsSub, showImage=False)


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)                              
    matches = bf.match(imgDescriptor, subDescriptor)                                
    matches = sorted(matches, key = lambda x:x.distance)                          
    numMatches = 10
    print(f"Matches = {len(matches)}")
    final_matches = matches[:numMatches]
    matchedImg, matchedSub = getMatchesKeypoints(final_matches, imgKeypoints, subKeypoints)

    # Drawing information
    cx, cy = int(width/2), int(height/2)
    rectangleCorners = rotate2DRectangle(x0, x1, y0, y1, cx, cy, -rotAngle)
    img = drawPolygon(img, rectangleCorners)

    subH, subW = sub.shape
    rectangleDrawOffset = 5
    sub = drawRectangle(sub, rectangleDrawOffset, rectangleDrawOffset, subW-rectangleDrawOffset, subH-rectangleDrawOffset)

    # Showing results
    inliers = countInliers(matchedImg, rectangleCorners)
    print(f"Total inliers = {inliers} - Ratio = {int(100*(inliers/len(final_matches)))}")
    result = cv2.drawMatches(img, imgKeypoints, sub, subKeypoints, final_matches, None, flags=2)
    result = resizeImage(result, 0.3)
    showImage(result)