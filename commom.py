import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from time import time
import random as rd

def showImage(img, title='Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def orbDetectorAndDescriptor(img, numFeatures, scaleFactor=1.2, nlevels=8, firstLevel=0, showImage=False):
    orb = cv2.ORB_create(numFeatures, scaleFactor=scaleFactor, nlevels=nlevels, firstLevel=firstLevel, WTA_K=2)
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

def imgSegmentation(img, nBlocks=(2,2)):
    horizontal = np.array_split(img, nBlocks[0])
    splitted_img = [np.array_split(block, nBlocks[1], axis=1) for block in horizontal]
   
    minHeigh, minWidth = 0, 0
    heighs, widths = [], []

    for row in range(nBlocks[0]):
        heighs.append(minHeigh)
        minHeigh += len(splitted_img[row][0])

    for col in range(nBlocks[1]):
        widths.append(minWidth)
        minWidth += len(splitted_img[0][col][0])

    minPoints = np.vstack((heighs, widths))
    return np.asarray(splitted_img, dtype=np.ndarray).reshape(nBlocks), minPoints

def revertImageSegmentation(imgArray, nBlocks=(2,2), title='Merged Image'):
    for h in range(nBlocks[0]):
        buffer = imgArray[h, 0]
        for w in range(1, nBlocks[1]):
            print(f"Img: {imgArray[h, w].shape} - Buffer: {buffer.shape}")
            buffer = np.hstack((buffer, imgArray[h, w]))
        if h == 0: result = buffer
        else: result = np.vstack((result, buffer))
    showImage(result, title = title)

def segmentedOrb(img, numFeatures, nBlocks=(2,2)):
    imgs, segmentationPoints = imgSegmentation(img, nBlocks=nBlocks)
    h, w = segmentationPoints
    result = []
    for row in range(imgs.shape[0]):
        for col in range(imgs.shape[1]):
            kp = orbDetector(imgs[row][col], numFeatures)
            print(row, col, len(kp))
            for p in kp:
                p.pt = (p.pt[0]+w[col], p.pt[1]+h[row])
            if len(result): result = np.hstack((result, kp))
            else: result = kp
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

def cannyEdgeDetection(img, title = 'Canny Edge Detector', showImg = True):     # TODO
    # https://www.youtube.com/watch?v=hUC1uoigH6s&list=PL2zRqk16wsdqXEMpHrc4Qnb5rA1Cylrhx&index=5
    pass
