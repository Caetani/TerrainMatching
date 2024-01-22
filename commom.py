import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from time import time
import random as rd
import rasterio
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from generalParameters import *
from numba import vectorize
import concurrent.futures
from scipy.stats import qmc
import gc
import sys
from memory_profiler import profile
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def showImage(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showImageGPU(img, title='Image'):
    imgDown = img.download()
    showImage(np.uint8(imgDown), title=title)

def scaledDepthMap(depth, color='terrain', saveas = False):
    ''' Avaiable colors:
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
    'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys',
    'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
    'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
    'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1',
    'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot',
    'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
    'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
    'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray',
    'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
    'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot',
    'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gr 'plasma_r', 'prism', 'prism_r', 'rainbow',
    'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
    'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain',
    'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
    'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    '''
    maxVal, minVal = np.max(depth), np.min(depth)
    fig, ax = plt.subplots()
    h = ax.imshow(depth, cmap = color)
    fig.colorbar(h)
    if saveas: fig.savefig(saveas)
    plt.show()

def drawKeypoints(img, kp, _showImage=False, size=5):
    imgWithKeypoints = cv2.drawKeypoints(img, kp, None, flags=0)
    if _showImage:
        showImage(imgWithKeypoints)
    return imgWithKeypoints

def drawCustomKeypoints(image, keypoints, size=0.5, thickness=0.1, color=(255, 0, 0)):
    # Create a blank image to draw keypoints
    output_img = np.copy(image)

    # Draw keypoints on the image
    h, w = image.shape
    p = max(h, w)
    size = int(size*p/100)
    thickness = int(thickness*p/100)

    for kp in keypoints:
        x, y = kp.pt
        '''attributes = dir(kp)
        for attribute in attributes:
            if not attribute.startswith("__"):
                print(attribute, "=", getattr(kp, attribute))
        exit()'''
        size = int(size)
        thickness = int(thickness)
        output_img = cv2.circle(output_img, (int(x), int(y)), size, color, thickness)
    
    return output_img

def generateResultImageString(stereo):
    # Commom parameters between all stereo algorithms in OpenCV.
    BS = stereo.getBlockSize()
    mx = stereo.getDisp12MaxDiff()
    min = stereo.getMinDisparity()
    num = stereo.getNumDisparities()
    rng = stereo.getSpeckleRange()
    wdw = stereo.getSpeckleWindowSize()

    if isinstance(stereo, cv2.StereoSGBM):
        md = stereo.getMode()
        p1 = stereo.getP1()
        p2 = stereo.getP2()
        cap = stereo.getPreFilterCap()
        UR = stereo.getUniquenessRatio()
        return f"BS_{BS}-mx_{mx}-min_{min}-num_{num}-rng_{rng}-wdw_{wdw}-md_{md}-p1_{p1}-p2_{p2}-cap_{cap}-UR_{UR}"

    return f"BS_{BS}-mx_{mx}-min_{min}-num_{num}-rng_{rng}-wdw_{wdw}"

def fastFeatureDetector(img, title='FAST Feature Detector', showImg=False, 
                        threshold=10, type=2, nonmaxSuppression=True):

    fast = cv2.FastFeatureDetector_create(threshold=threshold, type=type, nonmaxSuppression=nonmaxSuppression)
    keypoints = fast.detect(img, None)
    imgWithKeypoints = cv2.drawKeypoints(img, keypoints, None)

    print("Threshold: ", fast.getThreshold())
    print("nonmaxSuppression: ", fast.getNonmaxSuppression())
    print("neighborhood: ", fast.getType())
    print("Total Keypoints with nonmaxSuppression: ", len(keypoints))

    if showImg: showImage(imgWithKeypoints, title)
    return keypoints

def orbDetector(img, n):
    orb = cv2.ORB_create(n)
    kp = orb.detect(img, None)
    return kp

def orbDescriptor(img, kp):
    orb = cv2.ORB_create()
    kp, des = orb.compute(img, kp)
    return kp, des

def orbDetectorAndDescriptor(img, numFeatures, scaleFactor=1.2, nlevels=8, firstLevel=0, edgeThreshold=31, WTA_K=4, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20,  showImage=False):
    orb = cv2.ORB_create(nfeatures=numFeatures, scaleFactor=scaleFactor, nlevels=nlevels, firstLevel=firstLevel, WTA_K=WTA_K, edgeThreshold=edgeThreshold, scoreType=scoreType, patchSize=patchSize, fastThreshold=fastThreshold)
    kp, des = orb.detectAndCompute(img, None)
    if showImage:
        imgWithKeypoints = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        plt.imshow(imgWithKeypoints)
        plt.show()
    return kp, des

def orbImgWithKeypoints(img, n):
    '''
        Function created only to test the keypoints distribution
        after applying image segmentation.
    '''
    orb = cv2.ORB_create(n)
    kp, des = orb.detectAndCompute(img, None)
    imgWithKeypoints = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    return imgWithKeypoints

def imgSegmentation(img, nBlocks=(2, 2)):
    numRows, numCols = nBlocks
    horizontal = np.array_split(img, numRows)
    splitted_img = [np.array_split(block, numCols, axis=1) for block in horizontal]

    currentOffsetHeight = 0
    currentOffsetWidth = 0
    offsetHeights = []
    offsetWidths = []
    for row in range(numRows):
        offsetHeights.append(currentOffsetHeight)
        currentOffsetHeight += len(splitted_img[row][0])
    for col in range(numCols):
        offsetWidths.append(currentOffsetWidth)
        currentOffsetWidth += splitted_img[0][col].shape[1]
    images, y_offsets, x_offsets = [], [], []
    for row in range(numRows):
        for col in range(numCols):
            heightOffset = offsetHeights[row]
            widthOffset = offsetWidths[col]
            currentImg = splitted_img[row][col]
            images.append(currentImg)
            y_offsets.append(heightOffset)
            x_offsets.append(widthOffset)
    return images, y_offsets, x_offsets

    
def revertImageSegmentation(imgArray, nBlocks=(2,2), title='Merged Image'):
    for h in range(nBlocks[0]):
        buffer = imgArray[h, 0]
        for w in range(1, nBlocks[1]):
            print(f"Img: {imgArray[h, w].shape} - Buffer: {buffer.shape}")
            buffer = np.hstack((buffer, imgArray[h, w]))
        if h == 0: result = buffer
        else: result = np.vstack((result, buffer))
    showImage(result, title = title)

def FAST_GPU(gpuImg, numFeatures, startingThreshold=20, thresholdStep=1, minThreshold=4):     # TODO
    currentFeatures = 0
    startingThreshold = startingThreshold
    thresholdStep = thresholdStep
    minThreshold = minThreshold
    for t in range(startingThreshold, minThreshold-1, -thresholdStep):
        if t <= 0: threshold = 1
        else: threshold = t
        cudaFast = cv2.cuda.FastFeatureDetector.create(int(threshold), True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        cudaFast.setThreshold(int(threshold))
        kps = cudaFast.detectAsync(gpuImg)
        currentFeatures = kps.size()[0]
        if currentFeatures >= numFeatures:
            resultKps = cudaFast.convert(kps)
            sortedKps = sorted(resultKps, key=lambda x:x.response, reverse=True)
            
            # TODO Remove the line below:
            '''img = gpuImg.download()
            aux = drawCustomKeypoints(img, sortedKps)
            showImage(resizeImage(aux, 0.1))'''
            
            return sortedKps[:numFeatures]
    resultKps = cudaFast.convert(kps)
    sortedKps = sorted(resultKps, key=lambda x:x.response, reverse=True)
    return sortedKps


def FAST_CPU(img, numFeatures, startingThreshold=20, thresholdStep=1, minThreshold=2):
    currentFeatures = 0
    startingThreshold = startingThreshold
    thresholdStep = thresholdStep
    minThreshold = minThreshold
    for t in range(startingThreshold, minThreshold-1, -thresholdStep):
        threshold = max(1, int(t))
        fast = cv2.FastFeatureDetector.create(threshold=threshold, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        fast.setThreshold(int(threshold))
        kps = fast.detect(img)
        currentFeatures = len(kps)
        if currentFeatures >= numFeatures:
            sortedKps = sorted(kps, key=lambda x:x.response, reverse=True)
            return sortedKps[:numFeatures]
    sortedKps = sorted(kps, key=lambda x:x.response, reverse=True)
    #print(f"Final treshold: {threshold} - Number of kps: {currentFeatures}/{numFeatures}")
    return sortedKps

def FAST_MULTIPROCESSING(img, numFeatures, startingThreshold=20, thresholdStep=1, minThreshold=2):
    currentFeatures = 0
    startingThreshold = startingThreshold
    thresholdStep = thresholdStep
    minThreshold = minThreshold
    for t in range(startingThreshold, minThreshold-1, -thresholdStep):
        threshold = max(1, int(t))
        fast = cv2.FastFeatureDetector.create(threshold=threshold, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        fast.setThreshold(int(threshold))
        kps = fast.detect(img)
        currentFeatures = len(kps)
        if currentFeatures >= numFeatures: return True
        else: return False

def SEGMENTED_FAST_MULTIPROCESSING(img, numFeatures, startingThreshold=20, thresholdStep=1, minThreshold=2):
    currentFeatures = 0
    startingThreshold = startingThreshold
    thresholdStep = thresholdStep
    minThreshold = minThreshold
    for t in range(startingThreshold, minThreshold-1, -thresholdStep):
        threshold = max(1, int(t))
        fast = cv2.FastFeatureDetector.create(threshold=threshold, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
        fast.setThreshold(int(threshold))
        kps = fast.detect(img)
        currentFeatures = len(kps)
        if currentFeatures >= numFeatures:
            sortedKps = sorted(kps, key=lambda x:x.response, reverse=True)
            return sortedKps[:numFeatures]
    sortedKps = sorted(kps, key=lambda x:x.response, reverse=True)
    #print(f"Final treshold: {threshold} - Number of kps: {currentFeatures}/{numFeatures}")
    return sortedKps


def segmentedFAST_GPU(img, numFeatures, nBlocks=(2,2)): # TODO Update this function
    numRows, numCols = nBlocks
    
    horizontal = np.array_split(img, numRows)
    splitted_img = [np.array_split(block, numCols, axis=1) for block in horizontal]

    currentOffsetHeight = 0
    currentOffsetWidth = 0
    offsetHeights = []
    offsetWidths = []
    for row in range(numRows):
        offsetHeights.append(currentOffsetHeight)
        currentOffsetHeight += len(splitted_img[row][0])

    for col in range(numCols):
        offsetWidths.append(currentOffsetWidth)
        currentOffsetWidth += splitted_img[0][col].shape[1]

    #offsets = np.zeros((numRows, numCols, 2))
    result = []
    for row in range(numRows):
        for col in range(numCols):
            #offsets[row, col, :] = [offsetHeights[row], offsetWidths[col]]
            heightOffset = offsetHeights[row]
            widthOffset = offsetWidths[col]
            localH, localW = len(splitted_img[row][0]), splitted_img[0][col].shape[1]
            gpuImg = cv2.cuda.GpuMat(rows=localH, cols=localW, type=cv2.CV_8U)
            gpuImg.upload(splitted_img[row][col])
            keypoints = FAST_GPU(gpuImg, numFeatures)

            for kp in keypoints:
                #print(f"Before: {p.pt}")
                kp.pt = (kp.pt[0]+widthOffset, kp.pt[1]+heightOffset)
                #print(f"After: {p.pt}")
                result.append(kp)
            #if len(result): result = np.hstack((result, kp))
            #else: result = keypoints
    return result

def segmentedFAST_CPU(img, numFeatures, nBlocks=(2,2), startingThreshold=20, thresholdStep=1, minThreshold=4): # TODO Update this function
    numRows, numCols = nBlocks
    #FAST_CPU(gpuImg, numFeatures, startingThreshold=20, thresholdStep=1, minThreshold=4
    horizontal = np.array_split(img, numRows)
    splitted_img = [np.array_split(block, numCols, axis=1) for block in horizontal]

    currentOffsetHeight = 0
    currentOffsetWidth = 0
    offsetHeights = []
    offsetWidths = []
    for row in range(numRows):
        offsetHeights.append(currentOffsetHeight)
        currentOffsetHeight += len(splitted_img[row][0])
    for col in range(numCols):
        offsetWidths.append(currentOffsetWidth)
        currentOffsetWidth += splitted_img[0][col].shape[1]
    result = []
    for row in range(numRows):
        for col in range(numCols):
            heightOffset = offsetHeights[row]
            widthOffset = offsetWidths[col]
            currentImg = splitted_img[row][col]
            keypoints = FAST_CPU(currentImg, numFeatures, startingThreshold=startingThreshold,
                                 thresholdStep=thresholdStep, minThreshold=minThreshold)
            for kp in keypoints:
                kp.pt = (kp.pt[0]+widthOffset, kp.pt[1]+heightOffset)
                result.append(kp)
    return result

def segmentedORBDetect(img, numFeatures, nBlocks=(2,2)):
    numRows, numCols = nBlocks
    #FAST_CPU(gpuImg, numFeatures, startingThreshold=20, thresholdStep=1, minThreshold=4
    horizontal = np.array_split(img, numRows)
    splitted_img = [np.array_split(block, numCols, axis=1) for block in horizontal]

    currentOffsetHeight = 0
    currentOffsetWidth = 0
    offsetHeights = []
    offsetWidths = []
    for row in range(numRows):
        offsetHeights.append(currentOffsetHeight)
        currentOffsetHeight += len(splitted_img[row][0])
    for col in range(numCols):
        offsetWidths.append(currentOffsetWidth)
        currentOffsetWidth += splitted_img[0][col].shape[1]
    result = []
    for row in range(numRows):
        for col in range(numCols):
            heightOffset = offsetHeights[row]
            widthOffset = offsetWidths[col]
            currentImg = splitted_img[row][col]
            orbSegments = cv2.ORB.create(nfeatures=numFeatures, firstLevel=2, nlevels=5, fastThreshold=1)
            
            keypoints = orbSegments.detect(currentImg, None)
            for kp in keypoints:
                kp.pt = (kp.pt[0]+widthOffset, kp.pt[1]+heightOffset)
                result.append(kp)
    return result


def segmentedORB_GPU(img, numFeatures, nBlocks=(2, 2)):
    numRows, numCols = nBlocks
    
    horizontal = np.array_split(img, numRows)
    splitted_img = [np.array_split(block, numCols, axis=1) for block in horizontal]

    currentOffsetHeight = 0
    currentOffsetWidth = 0
    offsetHeights = []
    offsetWidths = []
    for row in range(numRows):
        offsetHeights.append(currentOffsetHeight)
        currentOffsetHeight += len(splitted_img[row][0])

    for col in range(numCols):
        offsetWidths.append(currentOffsetWidth)
        currentOffsetWidth += splitted_img[0][col].shape[1]

    #offsets = np.zeros((numRows, numCols, 2))
    result = []
    for row in range(numRows):
        for col in range(numCols):
            #offsets[row, col, :] = [offsetHeights[row], offsetWidths[col]]
            heightOffset = offsetHeights[row]
            widthOffset = offsetWidths[col]
            localH, localW = len(splitted_img[row][0]), splitted_img[0][col].shape[1]
            gpuImg = cv2.cuda.GpuMat(rows=localH, cols=localW, type=cv2.CV_8U)
            gpuImg.upload(splitted_img[row][col])
            keypoints = FAST_GPU(gpuImg, numFeatures)

            offsets = []
            descriptors = []
            
    return result

def applyLoG(img, blurWindowSize=(3, 3), type=cv2.CV_8U, _sigma=4):
    blur = cv2.GaussianBlur(img, blurWindowSize, sigmaX=_sigma, sigmaY=_sigma)
    laplacian = cv2.Laplacian(blur, type, ksize=3)
    return laplacian

def returnKeyPointArray(coordinatesArray, size=1):    # TODO
    keyPoints = []
    for x, y in coordinatesArray:
        keyPoints.append(cv2.KeyPoint(x=x, y=y, size=size))
    return keyPoints      

def logDetector(img, blurWindowSize=(3, 3), _showImage=False, type=cv2.CV_8U, sigma=2):
    img = applyLoG(img, blurWindowSize, type=type, _sigma=sigma)
    if _showImage:
        showImage(img)
    kp_zeros = np.argwhere(img>=10)
    kp_x, kp_y = kp_zeros[:, 1], kp_zeros[:, 0] # colunas, linhas
    kp = returnKeyPointArray(kp_x, kp_y)  
    return kp, drawKeypoints(img, kp=kp, _showImage=_showImage)

def getStereoImages(id, cameraMatrix, distCoeffs, grey=True):
    stereoImgSelection = {
    0: 'final_',
    1: 'corr_',
    2: 'small_',
    3: 'new_'
    }
    imgPrefix = stereoImgSelection[id]
    if grey:
        imgL = cv2.imread("Stereo Images/" + imgPrefix + 'left.jpg', cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread("Stereo Images/" + imgPrefix + 'right.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        imgL = cv2.imread("Stereo Images/" + imgPrefix + 'left.jpg')
        imgR = cv2.imread("Stereo Images/" + imgPrefix + 'right.jpg')
    imgL = cv2.undistort(imgL, cameraMatrix, distCoeffs, None)
    imgR = cv2.undistort(imgR, cameraMatrix, distCoeffs, None)
        
    assert imgL.shape == imgR.shape, "Images sizes do not match."
    return imgL, imgR, imgL.shape # shape = (height, width)

def resizeStereoImages(imgL, imgR, ratio, interpol=cv2.INTER_AREA):
    assert imgL.shape == imgR.shape, "Images sizes do not match."
    height, width = int(imgL.shape[0]*ratio), int(imgL.shape[1]*ratio)
    imgL = cv2.resize(imgL, (width, height), interpolation = cv2.INTER_AREA)
    imgR = cv2.resize(imgR, (width, height), interpolation = cv2.INTER_AREA)
    return imgL, imgR

def resizeImage(img, ratio, interpol=cv2.INTER_AREA):
    height, width = int(img.shape[0]*ratio), int(img.shape[1]*ratio)
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return img

def resizeImageGPU(img, ratio, interpol=cv2.INTER_AREA):
    width, height = img.size()
    width, height = int(width*ratio), int(height*ratio)
    img = cv2.cuda.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    return img

def getMatchesCoordinates(matches, leftKeypoints, rightKeypoints):
    leftCoordinates, rightCoordinates = [], []
    for match in matches:
        leftCoordinates.append(leftKeypoints[match.queryIdx].pt)
        rightCoordinates.append(rightKeypoints[match.trainIdx].pt)
    return np.array(leftCoordinates), np.array(rightCoordinates)

def getMatchesKeypoints(matches, leftKeypoints, rightKeypoints):
    matchedLeftKeypoints, matchedRigthKeypoints = [], []
    for match in matches:
        matchedLeftKeypoints.append(leftKeypoints[match.queryIdx])
        matchedRigthKeypoints.append(rightKeypoints[match.trainIdx])
    return matchedLeftKeypoints, matchedRigthKeypoints

def getMatchesKeypointsGPU(matches, leftKeypoints, rightKeypoints):
    print(f'Matches: {type(matches[0])}\nLeft keypoints: {type(leftKeypoints)}\nRight keypoints: {type(rightKeypoints)}')
    matchedLeftKeypoints, matchedRigthKeypoints = [], []
    for match in matches:
        matchedLeftKeypoints.append(leftKeypoints[match.queryIdx])
        matchedRigthKeypoints.append(rightKeypoints[match.trainIdx])
    return matchedLeftKeypoints, matchedRigthKeypoints

def inliersRatio(inliersArray):
    return np.count_nonzero(inliersArray) / len(inliersArray)

def rectification(img1, img2, pts1, pts2, F):
    """This function is used to rectify the images to make camera pose's parallel and thus make epiplines as horizontal.
        Since camera distortion parameters are not given we will use cv2.stereoRectifyUncalibrated(), instead of stereoRectify().
    """

    # Stereo rectification
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1))
    print("H1",H1)
    print("H2",H2)

    rectified_pts1 = np.zeros((pts1.shape), dtype=int)
    rectified_pts2 = np.zeros((pts2.shape), dtype=int)

    # Rectify the feature points
    for i in range(pts1.shape[0]):
        source1 = np.array([pts1[i][0], pts1[i][1], 1])
        new_point1 = np.dot(H1, source1)
        new_point1[0] = int(new_point1[0]/new_point1[2])
        new_point1[1] = int(new_point1[1]/new_point1[2])
        new_point1 = np.delete(new_point1, 2)
        rectified_pts1[i] = new_point1

        source2 = np.array([pts2[i][0], pts2[i][1], 1])
        new_point2 = np.dot(H2, source2)
        new_point2[0] = int(new_point2[0]/new_point2[2])
        new_point2[1] = int(new_point2[1]/new_point2[2])
        new_point2 = np.delete(new_point2, 2)
        rectified_pts2[i] = new_point2

    # Rectify the images and save them
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    
    cv2.imwrite("rectified_1.png", img1_rectified)
    cv2.imwrite("rectified_2.png", img2_rectified)
    
    return rectified_pts1, rectified_pts2, img1_rectified, img2_rectified

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    """This fucntion is used to visualize the epilines on the images
        img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 5)
        img1color = cv2.circle(img1color, (int(pt1[0]), int(pt1[1])), 15, color, -1)
        img2color = cv2.circle(img2color, (int(pt2[0]), int(pt2[1])), 15, color, -1)
    
    return img1color, img2color

def evaluateStereoRactification(imgLeft, imgRight, numMatches):
    initLeft, initRight = imgLeft, imgRight

    numFeatures = 10*numMatches
    leftKeypoints, leftDescriptor = orbDetectorAndDescriptor(imgLeft, numFeatures, showImage=False)
    rightKeypoints, rightDescriptor = orbDetectorAndDescriptor(imgRight, numFeatures, showImage=False)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)                              
    matches = bf.match(leftDescriptor, rightDescriptor)                                
    matches = sorted(matches, key = lambda x:x.distance)
    final_matches = matches[:numMatches]                                               

    ptsLeft, ptsRight = getMatchesCoordinates(final_matches, leftKeypoints, rightKeypoints)
    F, inliers = cv2.findFundamentalMat(ptsLeft, ptsRight, cv2.FM_LMEDS)
    ptsLeft = ptsLeft[inliers.ravel() == 1]
    ptsRight = ptsRight[inliers.ravel() == 1]

    linesLeft = cv2.computeCorrespondEpilines(ptsRight.reshape(-1, 1, 2), 2, F)
    linesLeft = linesLeft.reshape(-1, 3)
    #img5, img6 = drawlines(imgLeft, imgRight, linesLeft, ptsLeft, ptsRight)
    img5, img6 = drawlines(initLeft, initRight, linesLeft, ptsLeft, ptsRight)
    
    # Find epilines corresponding to 
    # points in left image (first image) and
    # drawing its lines on right image
    linesRight = cv2.computeCorrespondEpilines(ptsLeft.reshape(-1, 1, 2),  1, F)
    linesRight = linesRight.reshape(-1, 3)
    
    #img3, img4 = drawlines(imgRight, imgLeft, linesRight, ptsRight, ptsLeft)
    img3, img4 = drawlines(initRight, initLeft, linesRight, ptsRight, ptsLeft)
    
    result = np.hstack((img5, img3))
    plt.imshow(result)
    #plt.subplot(121), plt.imshow(img5)
    #plt.subplot(122), plt.imshow(img3)
    plt.show()

def getMap(fileName, dtype=np.uint8):
    with rasterio.open(fileName, 'r') as ds:
        img = ds.read()[0]  # read all raster values
    return np.matrix(img, dtype=dtype)

def cutSubregion(map, x0, y0, x1, y1):
    return map[int(y0):int(y1), int(x0):int(x1)]

def cutSubregionGPU(map, x0, y0, x1, y1):
    croppedMap = cv2.cuda_GpuMat(map, [y0, y1], [x0, x1])
    return croppedMap

def applyNoise(img, std, dtype=np.float32):
    h, w = img.shape
    noisyImg = np.random.randn(h, w)
    noisyImg = np.float32(noisyImg*std + img)
    assert noisyImg.dtype == dtype, f"In applyNoise function: Expected dtype {dtype}, result is {noisyImg.dtype}."
    return noisyImg

def applyNoiseGPU(img, std=1):
    w, h = img.size()
    np_img = img.download()
    noisyImg = np.random.randn(h, w)
    noisyImg = noisyImg*std + np_img
    #noisyImg = np.uint8(noisyImg)
    noisyImg = np.float32(noisyImg)
    result = cv2.cuda_GpuMat(noisyImg)
    return result

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

def rotateMapGPU(map, deg):
    width, height = map.size()
    cr = (width/2,height/2)     # define center of rotation
    M = cv2.getRotationMatrix2D(cr,deg,1)    # 'deg' in anticlockwise direction
    return cv2.cuda.warpAffine(map,M,(width,height))  # apply warpAffine() method to perform image rotation

def plotSideBySide(data1, data2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(data1)
    ax2.plot(data2)
    plt.grid()
    plt.plot()

def rotate2DRectangle(x0, x1, y0, y1, cx, cy, deg):
    points = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
    return rotateAroundCenter(points, cx, cy, deg)
     
def drawPolygon(img, pts, color=(255, 0, 0)):
    img = np.int32(img)
    height, width = img.shape
    if width >= height: max = width
    else:  max = height
    thickness = int(max*0.003)
    img = cv2.polylines(img, np.int32([pts]), 1, color=color, thickness=thickness)
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

def calcInitialLevel(imgShape, scaleFactor, desiredInitialArea):
    A = imgShape[0]*imgShape[1]
    return int(math.log(desiredInitialArea/A, scaleFactor)/2)

def calcNumberLevels(imgShape, scaleFactor, initialLevel, desiredFinalArea):
    '''Used to calculate pyramid parameters in
    ORB Detector.'''
    A = imgShape[0]*imgShape[1]
    return int(math.log(A/desiredFinalArea, scaleFactor)/2 + initialLevel)

def performanceData(hist, title):
    title = f"\n{'='*10} {title} {'='*10}"
    print(title)
    mean, std = np.mean(hist), np.std(hist)
    print(f" Média = {mean}\n Std dev = {std}\n{len(title)*'='}\n") 

    return mean, std

def getCoord(kp):
    x, y = kp.pt[0], kp.pt[1]
    return x, y

def countSuccessfullMatches(final_matches, imgKeypoints, subKeypoints, resizeScale, x0, y0, cx, cy, rotAngle, thresholdRadius=5, showData=False):
    '''
    Function used to calculate how many matches were truly successful.
    
    It verifies that the distance [px] between real point (in greater map)
    and predicted point (sub-region) is smaller than thresholdRadius [px]

    x0, y0: used to calculate sub-region position in greater map.
    cx, cy: center of the greater map (width/2, height/2).
    rotAngle: the angle that greater map was rotated to extract sub-region.
    '''
    matchedImg, matchedSub = getMatchesKeypoints(final_matches, imgKeypoints, subKeypoints)
    counter = 0
    for i in range(len(final_matches)):
        img_x, img_y = getCoord(matchedImg[i])
        sub_x, sub_y = getCoord(matchedSub[i])
        sub_x, sub_y = int(sub_x*resizeScale + x0), int(sub_y*resizeScale + y0)
        [[sub_x, sub_y]] = rotateAroundCenter([[sub_x, sub_y]], cx, cy, -rotAngle)
        delta_x, delta_y = abs(img_x-sub_x), abs(img_y-sub_y)
        if showData: print("X: ", int(delta_x), "y: ", int(delta_y))
        if math.sqrt(delta_x**2 + delta_y**2) <= thresholdRadius: counter += 1
    return counter

def computeScore(final_matches, imgKeypoints, subKeypoints, resizeScale, x0, y0, cx, cy, rotAngle, thresholdRadius=5):
    '''
    Function used to calculate how many matches were truly successful.
    
    It verifies that the distance [px] between real point (in greater map)
    and predicted point (sub-region) is smaller than thresholdRadius [px]

    x0, y0: used to calculate sub-region position in greater map.
    cx, cy: center of the greater map (width/2, height/2).
    rotAngle: the angle that greater map was rotated to extract sub-region.
    '''
    matchedImg, matchedSub = getMatchesKeypoints(final_matches, imgKeypoints, subKeypoints)
    score, inliers = 0, 0
    bestMatches = 0
    for i in range(len(final_matches)):
        img_x, img_y = getCoord(matchedImg[i])
        sub_x, sub_y = getCoord(matchedSub[i])
        sub_x, sub_y = int(sub_x*resizeScale + x0), int(sub_y*resizeScale + y0)
        [[sub_x, sub_y]] = rotateAroundCenter([[sub_x, sub_y]], cx, cy, -rotAngle)
        delta_x, delta_y = abs(img_x-sub_x), abs(img_y-sub_y)

        bValidPoint = math.sqrt(delta_x**2 + delta_y**2) <= thresholdRadius
        distance = final_matches[i].distance
        score += bValidPoint*256 - distance
        inliers += bValidPoint
        if i < 10 and bValidPoint: bestMatches += 1
    #return score, inliers
    return score, inliers, bestMatches

def computeMatchesScore(final_matches, imgKeypoints, subKeypoints, resizeScale, x0, y0, cx, cy, rotAngle, thresholdRadius, matcherType, pitch, roll, numBestMatches, bestMatchesMultiplier=3):
    # TODO Melhor função objetivo até agora
    '''
    Function used to calculate how many matches were truly successful.
    
    It verifies that the distance [px] between real point (in greater map)
    and predicted point (sub-region) is smaller than thresholdRadius [px]

    x0, y0: used to calculate sub-region position in greater map.
    cx, cy: center of the greater map (width/2, height/2).
    rotAngle: the angle that greater map was rotated to extract sub-region.
    pitch/roll: angles applied in sub-region distortion.
    '''
    matchedImg, matchedSub = getMatchesKeypoints(final_matches, imgKeypoints, subKeypoints)
    score, inliers, bestMatches = 0, 0, 0
    cx, cy = round(cx/resizeScale), round(cy/resizeScale)
    for i in range(len(final_matches)):
        img_x, img_y = getCoord(matchedImg[i])
        img_x, img_y = round(img_x/resizeScale), round(img_y/resizeScale)
        sub_x, sub_y = getCoord(matchedSub[i])
        sub_x, sub_y = sub_x / math.cos(math.radians(roll)), sub_y / math.cos(math.radians(pitch))
        sub_x, sub_y = round(sub_x + x0/resizeScale), round(sub_y + y0/resizeScale)
        [[sub_x, sub_y]] = rotateAroundCenter([[sub_x, sub_y]], cx, cy, -rotAngle)
        delta_x, delta_y = abs(img_x-sub_x), abs(img_y-sub_y)
        bValidPoint = math.floor(math.sqrt(delta_x**2 + delta_y**2)) <= thresholdRadius
        distance = final_matches[i].distance
        if matcherType == cv2.NORM_HAMMING:
            currentScore = (bValidPoint*(256-distance))**2
        elif matcherType == cv2.NORM_HAMMING2:
            currentScore = bValidPoint*(65536-distance)
        inliers += bValidPoint
        if i < numBestMatches and bValidPoint:
            bestMatches += 1
            currentScore *= bestMatchesMultiplier
        score += currentScore
    return score, inliers, bestMatches

def minimizeAllMatches(final_matches, imgKeypoints, subKeypoints, resizeScale, x0, y0, cx, cy, rotAngle, thresholdRadius=5):
    # TODO Muito ruim essa função objetivo
    '''
    Function used to calculate how many matches were truly successful.
    
    It verifies that the distance [px] between real point (in greater map)
    and predicted point (sub-region) is smaller than thresholdRadius [px]

    x0, y0: used to calculate sub-region position in greater map.
    cx, cy: center of the greater map (width/2, height/2).
    rotAngle: the angle that greater map was rotated to extract sub-region.
    '''
    matchedImg, matchedSub = getMatchesKeypoints(final_matches, imgKeypoints, subKeypoints)
    score, inliers = 0, 0
    bestMatches = 0
    for i in range(len(final_matches)):
        img_x, img_y = getCoord(matchedImg[i])
        sub_x, sub_y = getCoord(matchedSub[i])
        sub_x, sub_y = int(sub_x*resizeScale + x0), int(sub_y*resizeScale + y0)
        [[sub_x, sub_y]] = rotateAroundCenter([[sub_x, sub_y]], cx, cy, -rotAngle)
        delta_x, delta_y = abs(img_x-sub_x), abs(img_y-sub_y)

        bValidPoint = math.sqrt(delta_x**2 + delta_y**2) <= thresholdRadius
        distance = final_matches[i].distance
        score += 256 - distance
        inliers += bValidPoint
        if i < 10 and bValidPoint: bestMatches += 1
    #return score, inliers
    return score, inliers, bestMatches

def applyPlaneDistortion(img, pitch, roll, z0=0, interpolationScale=5, dtype=np.float32):
    y0 = -math.tan(math.radians(pitch))
    x0 = -math.tan(math.radians(roll))
    h, w = img.shape
    scale = interpolationScale
    img = cv2.resize(img, (w*scale, h*scale), cv2.INTER_CUBIC)  # INTER_CUBIC to enlarge image
    x0, y0 = x0/scale, y0/scale
    plane = np.fromfunction(lambda y, x: y0*y + x0*x, img.shape, dtype=dtype)
    result = plane + img
    result = cv2.resize(result, (int(w*math.cos(math.radians(roll))), int(h*math.cos(math.radians(pitch)))), cv2.INTER_AREA) # INTER_AREA to shrink an image.
    assert result.dtype == dtype, "dtype missmatch in applyPlaneDistortion function calculation."
    return result

def applyPlaneDistortionGPU(img, pitch, roll, z0=0, interpolationScale=5):
    y0 = -math.tan(math.radians(pitch))
    x0 = -math.tan(math.radians(roll))
    w, h = img.size()
    scale = interpolationScale
    img = cv2.cuda.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)  # INTER_CUBIC to enlarge image
    x0, y0 = x0/scale, y0/scale
    scaled_w, scaled_h = img.size()
    plane_np = np.fromfunction(lambda y, x: y0*y + x0*x, (scaled_h, scaled_w), dtype=float)
    img_np = img.download()
    img_np = img_np + plane_np
    result = cv2.resize(img_np, (w, h), cv2.INTER_AREA) # INTER_AREA to shrink an image.
    result = cv2.cuda_GpuMat(img_np)
    return result

def fitImage_UINT8(img):
    '''
    Fit image in uint8 (0, 255, 1) format.
    '''
    minValue = np.min(img)
    maxValue = np.max(img)
    delta = maxValue-minValue
    img = 255*((img-minValue)/delta)
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return img

@vectorize(["uint8(float32, float32, float32)"], target='cuda')
def float32_to_uint8_GPU(img, min, max):
    delta = max-min
    img = round(255*((img-min)/delta))
    return img

def float32_to_uint8_CPU(img, min, max):
    '''
    Fit image in uint8 (0, 255, 1) format.
    '''
    delta = max-min
    img = np.round(255*((img-min)/delta)).astype(np.uint8)
    return img

def calculateNumberOfBlocks(lenghtInPixels, img):
    h, w = img.shape
    nBlocksY = round(h/lenghtInPixels)
    nBlocksX = round(w/lenghtInPixels)
    return (nBlocksY, nBlocksX)

def getBoundaries(spacesArr):
    '''
        Get lower and upper boundary values for Latin Hypercube.
    '''
    lowerBoundaries, upperBoundaries = [], []
    lowerBoundaries = np.zeros(len(spacesArr))
    for space in spacesArr:
        upper = len(space)-1
        upperBoundaries.append(upper)
    return lowerBoundaries, upperBoundaries

def generatePopulation(numPop, SPACE):
    lowerBoundaries, upperBoundaries = getBoundaries(SPACE)
    sampler = qmc.LatinHypercube(d=len(SPACE))
    initialPopulation = sampler.random(n=numPop)
    initialPopulation = qmc.scale(initialPopulation, lowerBoundaries, upperBoundaries)
    for individual in initialPopulation:
        for i in range(len(individual)):
            individual[i] = SPACE[i][round(individual[i])]
    return initialPopulation

def listReferences(object):
    references = list(gc.get_referrers(object))
    for i, reference in references:
        print(f"Reference {i}: {reference}")
    print(f"\tReference count: {sys.getrefcount(object)}")

@vectorize(["float32(float32, float32)"], target='cuda')       
def calculate_mape_vectorized(map_val, noisy_map_val):
    """
    Vectorized function to calculate Mean Absolute Percentage Error (MAPE) between two values.

    Parameters:
    - map_val: Value from the original image.
    - noisy_map_val: Value from the noisy image.

    Returns:
    - mape: Mean Absolute Percentage Error.
    """
    return abs((map_val - noisy_map_val) / (map_val + 1e-8)) * 100.0

def calculate_mape(map_data, noisy_map_data):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between two images.

    Parameters:
    - map_data: 2D NumPy array representing the original image.
    - noisy_map_data: 2D NumPy array representing the noisy image.

    Returns:
    - mape: Mean Absolute Percentage Error.
    """
    # Ensure images have the same shape
    assert map_data.shape == noisy_map_data.shape, "Images must have the same shape."

    # Use the vectorized function directly on the arrays
    abs_percentage_error = calculate_mape_vectorized(map_data, noisy_map_data)

    # Calculate mean absolute percentage error
    total_pixels = map_data.size
    mape_sum = np.sum(abs_percentage_error)

    mape = mape_sum / total_pixels

    return mape

@vectorize(["float32(float32, float32)"], target='cuda')   
def calculate_rmse_vectorized(map_val, noisy_map_val):
    """
    Vectorized function to calculate squared error between two values.

    Parameters:
    - map_val: Value from the original image.
    - noisy_map_val: Value from the noisy image.

    Returns:
    - squared_error: Squared error.
    """
    return (map_val - noisy_map_val)**2

def calculate_rmse(map_data, noisy_map_data):
    """
    Calculate Root Mean Squared Error (RMSE) between two images.

    Parameters:
    - map_data: 2D NumPy array representing the original image.
    - noisy_map_data: 2D NumPy array representing the noisy image.

    Returns:
    - rmse: Root Mean Squared Error.
    """
    # Ensure images have the same shape
    assert map_data.shape == noisy_map_data.shape, "Images must have the same shape."

    # Use the vectorized function directly on the arrays
    squared_error = calculate_rmse_vectorized(map_data.flatten(), noisy_map_data.flatten())

    # Calculate mean squared error
    mse = np.mean(squared_error)

    # Calculate RMSE
    rmse = np.sqrt(mse)

    return rmse
