'''
    Script with commmom definitions of terrain matching algorithm for oprimization.
'''

import numpy as np
import pandas as pd
import cv2

files = [r'Datasets\RS\merged_map.tif',
         r'Datasets\Cyprus\Cyprus_data.tif',
         r'Datasets\USGS\OK_Panhandle.tif']

datasetsResolutions = [30, 5, 1]

SPACE_sigmaImg                    = np.arange(1, 7+1, 1)
SPACE_sigmaSub                    = np.arange(1, 7+1, 1)
SPACE_windowSizeScaleImg          = np.arange(1, 10+1, 1)
SPACE_windowSizeScaleSub          = np.arange(1, 10+1, 1)
SPACE_patchSize                   = np.arange(11, 51+1, 10)
SPACE_nLevelsSub                  = np.arange(4, 12+1, 1)
SPACE_initLevelSub                = np.arange(0, 3+1, 1)
SPACE_scaleFactor                 = np.arange(1.1, 1.6+0.1, 0.1)
SPACE_fastThreshold               = np.arange(5, 45+1, 5)
SPACE_fastBlockSizeInPixels       = np.arange(100, 1_000+100, 100)
SPACE_fastFeatureDensity          = np.arange(0.1, 0.5+0.1, 0.1)
#SPACE_fastFeatureDensity          = np.arange(0.01, 0.1+0.05, 0.05)
SPACE_fastMinThreshold            = np.arange(4, 10+3, 3)
#SPACE_fastMinThreshold            = np.arange(5, 15+5, 5)
SPACE_WTA                         = np.arange(2, 4+1, 1)    # cv2.NORM_HAMMING (max 256) or cv2.NORM_HAMMING2 (max 65.536)
SPACE_crossCheck                  = np.arange(0, 1+1, 1)    # True or False
SPACE_ALL = [SPACE_sigmaImg, SPACE_sigmaSub, SPACE_windowSizeScaleImg,
             SPACE_windowSizeScaleSub, SPACE_patchSize, SPACE_nLevelsSub,
             SPACE_initLevelSub, SPACE_scaleFactor, SPACE_fastThreshold,
             SPACE_fastBlockSizeInPixels, SPACE_fastFeatureDensity,
             SPACE_fastMinThreshold, SPACE_WTA, SPACE_crossCheck]

GA_num_parents_mating             = np.arange(0.4, 0.8+0.2, 0.2)
GA_parent_selection_type          = np.arange(0, 5+1, 1)
GA_keep_elitism                   = np.arange(0, 2+1, 1)
#GA_keep_elitism                   = np.arange(1, 2+1, 1)
GA_crossover_type                 = np.arange(0, 3+1, 1)
GA_crossover_probability          = np.arange(0.4, 1+0.3, 0.3)
#GA_mutation_type                  = np.arange(0, 4+1, 1)
GA_mutation_type                  = np.arange(0, 0+1, 1)
GA_mutation_num_genes_init        = np.arange(0.1, 0.5+0.1, 0.2)
GA_mutation_num_genes_end         = np.arange(0.2, 1+0.1, 0.4)
GA_PARAMETERS_SPACES = [GA_num_parents_mating, GA_parent_selection_type,
                 GA_keep_elitism, GA_crossover_type, GA_crossover_probability,
                 GA_mutation_num_genes_init, GA_mutation_num_genes_end]#GA_mutation_type,
                 
def GA_getCrossoverType(n):
    crossoverTypeDict = {
        0: "single_point",
        1: "two_points",
        2: "uniform",
        3: "scattered",
    }
    return crossoverTypeDict[n]

def GA_getParentSelectionType(n):
    parentSelectionDict = {
        0: "sss",           # steady-state selection
        1: "rws",           # roulette wheel selection
        2: "sus",           # stochastic universal selection
        3: "rank",          # rank selection
        4: "random",        # random selection
        5: "tournament"     # tournament selection
    }
    return parentSelectionDict[n]

def printGAParameters(GA_parameters, populationSize):
    pct_num_parents_mating = GA_parameters[0]
    parent_selection_type  = GA_parameters[1]
    keep_elitism           = GA_parameters[2]
    crossover_type         = GA_parameters[3]
    crossover_probability  = GA_parameters[4]
    pctAdaptiveInit        = GA_parameters[5]
    pctAdaptiveEnd         = GA_parameters[6]


    parent_selection_type = GA_getParentSelectionType(parent_selection_type)
    crossover_type        = GA_getCrossoverType(crossover_type)
    mutation_type         = "adaptive"
    keep_elitism          = round(keep_elitism)
    crossover_probability = round(crossover_probability, 2)
    num_parents_mating    = round(pct_num_parents_mating*populationSize, 3)
    mutation_num_genes    = (max(round(pctAdaptiveInit*populationSize), 1), max(round(pctAdaptiveEnd*pctAdaptiveInit*populationSize), 1))

    print(f"\nparent_selection_type: {parent_selection_type}\n"
          f"\tcrossover_type: {crossover_type}\n"
          f"\tmutation_type: {mutation_type}\n"
          f"\tkeep_elitism: {keep_elitism}\n"
          f"\tcrossover_probability: {crossover_probability}\n"
          f"\tnum_parents_mating: {num_parents_mating}\n"
          f"\tmutation_num_genes: {mutation_num_genes}")
    return

def printCromossome(cromossome):
    sigmaImg                    = cromossome[0]
    sigmaSub                    = cromossome[1]
    windowSizeScaleImg          = cromossome[2]
    windowSizeScaleSub          = cromossome[3]
    patchSize                   = cromossome[4]
    nLevelsSub                  = cromossome[5]
    initLevelSub                = cromossome[6]
    scaleFactor                 = cromossome[7]
    fastThreshold               = cromossome[8]
    fastBlockSizeInPixels       = cromossome[9]
    fastFeatureDensity          = cromossome[10]
    fastMinThreshold            = cromossome[11]
    WTA                         = cromossome[12]
    crossCheck                  = cromossome[13]
    patchSize             = int(patchSize)
    nLevelsSub            = int(nLevelsSub)
    initLevelSub          = int(initLevelSub)
    scaleFactor           = round(scaleFactor, 2)
    fastThreshold         = int(fastThreshold)
    fastBlockSizeInPixels = int(fastBlockSizeInPixels)
    fastFeatureDensity    = round(fastFeatureDensity, 1)
    fastMinThreshold      = int(fastMinThreshold)
    WTA                   = int(WTA)
    crossCheck            = bool(crossCheck)
    print(f"\n\tsigmaImg: {sigmaImg}\n\tsigmaSub: {sigmaSub}\n\twindowSizeScaleImg: {windowSizeScaleImg}\n\twindowSizeScaleSub: {windowSizeScaleSub}\n\tpatchSize: {patchSize}\n\tnLevelsSub: {nLevelsSub}\n\tinitLevelSub: {initLevelSub}\n\tscaleFactor: {scaleFactor}\n\tfastThreshold: {fastThreshold}\n\tfastBlockSizeInPixels: {fastBlockSizeInPixels}\n\tfastFeatureDensity: {fastFeatureDensity}\n\tfastMinThreshold: {fastMinThreshold}\n\tWTA: {WTA}\n\tcrossCheck: {crossCheck}")
    return

def printSample(sample):
    pitch        = sample[0]
    roll         = sample[1]
    x0           = sample[2]
    y0           = sample[3]
    subSize      = sample[4]
    rotAngle     = sample[5]
    resizeScale  = sample[6]
    noise        = sample[7]
    fileNumber   = sample[8]

    pitch       = round(pitch, 1)
    roll        = round(roll, 1)
    x0, y0      = round(x0, 2), round(y0, 2)
    subSize     = round(subSize, 2)
    rotAngle    = round(rotAngle, 1)
    resizeScale = round(resizeScale, 2)
    noise       = round(noise, 1)
    fileNumber  = int(fileNumber)

    file = files[fileNumber]

    print(f"\n\tPitch: {pitch}\n\tRoll: {roll}\n\tx0, y0: ({x0}, {y0})\n\tsubSize: {subSize}\n\trotAngle: {rotAngle}\n\tresizeScale: {resizeScale}\n\tNoise: {noise}\n\tFile: {file}")
    return
    
def getMatcherType(WTA):
    return cv2.NORM_HAMMING*(WTA == 2) + cv2.NORM_HAMMING2*(WTA > 2)

def calculateNumFastFeatures(fastFeatureDensity, fastBlockSizeInPixels):
    return int(fastFeatureDensity*fastBlockSizeInPixels)

def correctData(number, column):
    if column == "sigmaImg" or column == "sigmaSub" or column == 'windowSizeScaleImg' or column == 'windowSizeScaleSub' or column == 'nLevelsImg' or column == 'nLevelsSub' or column == 'initLevelImg' or column == 'initLevelSub' or column == 'ratioImgAndSubFeatureNumber':
        return round(number)
    elif column == "scaleFactor":
        return round(number, 1)
    elif column == "patchSize" or column == "fastThreshold":
        number = round(number)
        if number % 2 == 0:
            return number + 1
        else: return number

def closestIndex(n, numArray):
    # Calcula as diferenças absolutas entre o número 'n' e os elementos do vetor 'numArray'
    diferenca_absoluta = np.abs(np.array(numArray) - n)
    
    # Encontra o índice do número mais próximo
    indice_numero_proximo = diferenca_absoluta.argmin()
    
    return indice_numero_proximo

def applyNoiseInCromossome(cromossome):
    numGenes = len(SPACE_ALL)
    numChangedGenes = int(abs(np.random.normal())) + 2
    selectedPositions = []
    while len(selectedPositions) <= numChangedGenes:
        posRandom = int(numGenes*np.random.random())
        if not posRandom in selectedPositions:
            selectedPositions.append(posRandom)

    for genePosition in selectedPositions:
        try:
            elementPosition = np.where(np.isclose(SPACE_ALL[genePosition], cromossome[genePosition]))[0][0]
        except:
            elementPosition = closestIndex(cromossome[genePosition], SPACE_ALL[genePosition])
        randomNumber = int(abs(np.random.normal()))
        if elementPosition == len(SPACE_ALL[genePosition])-1:
            elementPosition -= 1+randomNumber
        elif elementPosition == 0:
            elementPosition += 1+randomNumber
        else:
            elementPosition += (1+(np.random.rand() > 0.5)*-2)*(1+randomNumber)
            elementPosition = min(elementPosition, len(SPACE_ALL[genePosition])-1)
            elementPosition = max(elementPosition, 0)
    cromossome[genePosition] = SPACE_ALL[genePosition][elementPosition]
    return cromossome

def generateCentroidPopulation(popSize):
    popSize = int(popSize)
    maxClusters = 15
    filePreffix = 'Análises estatísticas\Cluster\Centroids\centroid_k_'
    fileSuffix = '.xlsx'
    remaining = popSize
    clusterNumberArray = []
    while remaining > 0:
        subtract = min(remaining, maxClusters)
        remaining -= subtract
        clusterNumberArray.append(subtract)

    populationArray = []
    for clusterNumber in clusterNumberArray:
        data = pd.read_excel(filePreffix + f'{clusterNumber}' + fileSuffix)
        columns = data.columns
        data = data.to_numpy()
        for i in range(len(data)):
            populationArray.append(data[i][:])
    
    for cromossome in populationArray:
        for geneIndex, col in enumerate(columns):
            cromossome[geneIndex] = correctData(cromossome[geneIndex], col)
        cromossome = applyNoiseInCromossome(cromossome)
    return populationArray

def selectSample(sampleId, pitch=0, roll=0, x0=0.4, y0=0.4, subSize=0.1, rotAngle=0, resizeScale=0.8, noise=1, fileNumber=0):
    if sampleId == '9_samples_with_limits':
        #pitch,roll,x0,y0,subSize,rotAngle,resizeScale,noise, fileNumber
        sample_0 = [0.0, 3.0, 0.45, 0.6, 0.15, 50.0, 0.8, 1.5, 1.0]
        sample_1 = [3.0, 2.0, 0.6, 0.35, 0.15, 100.0, 0.4, 1.0, 0.0]
        sample_2 = [4.0, 1.0, 0.35, 0.5, 0.1, 220.0, 0.5, 2.0, 2.0]
        sample_3 = [5.0, 4.0, 0.5, 0.4, 0.05, 30.0, 0.7, 1.5, 1.0]
        sample_4 = [2.0, 1.0, 0.55, 0.55, 0.2, 350.0, 0.7, 1.0, 1.0]
        sample_5 = [2.0, 3.0, 0.4, 0.55, 0.15, 280.0, 0.6, 0.5, 1.0]
        sample_6 = [4.0, 5.0, 0.7, 0.3, 0.05, 180.0, 0.9, 2.0, 2.0]
        sample_7 = [2.0, 3.0, 0.35, 0.7, 0.15, 260.0, 0.3, 1.0, 0.0]
        sample_8 = [1.0, 0.0, 0.6, 0.45, 0.1, 150.0, 1.0, 1.5, 0.0]

        samples = [sample_0, sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8]
        return samples

    if sampleId == '9_samples':
        #pitch,roll,x0,y0,subSize,rotAngle,resizeScale,noise, fileNumber
        sample_0 = [4.0, 3.0, 0.55, 0.65, 0.05, 20.0, 0.6, 1.5, 1.0]
        sample_1 = [1.0, 9.0, 0.3, 0.4, 0.2, 260.0, 0.7, 2.0, 2.0]
        sample_2 = [6.0, 8.0, 0.45, 0.35, 0.15, 170.0, 0.8, 0.5, 0.0]
        sample_3 = [9.0, 1.0, 0.65, 0.35, 0.15, 310.0, 0.8, 1.0, 1.0]
        sample_4 = [1.0, 6.0, 0.7, 0.6, 0.15, 120.0, 0.4, 1.0, 2.0]
        sample_5 = [3.0, 2.0, 0.35, 0.5, 0.1, 310.0, 0.9, 0.5, 0.0]
        sample_6 = [8.0, 7.0, 0.5, 0.5, 0.15, 80.0, 0.3, 1.5, 2.0]
        sample_7 = [5.0, 5.0, 0.55, 0.65, 0.1, 50.0, 0.8, 1.5, 1.0]
        sample_8 = [7.0, 4.0, 0.4, 0.45, 0.05, 190.0, 0.5, 1.5, 0.0]

        samples = [sample_0, sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8]
        return samples
        
    if sampleId == '9_samples_old':     # resizeScale > 1 --> Muito pesado
        sample_0 = [9.0, 7.0, 0.3, 0.35, 0.15, 230.0, 0.6, 2.0, 1.0]
        sample_1 = [1.0, 9.0, 0.55, 0.65, 0.05, 280.0, 1.4, 1.5, 2.0]
        sample_2 = [7.0, 6.0, 0.65, 0.55, 0.1, 340.0, 1.5, 1.5, 0.0]
        sample_3 = [6.0, 4.0, 0.7, 0.4, 0.15, 160.0, 0.5, 1.5, 1.0]
        sample_4 = [5.0, 2.0, 0.4, 0.5, 0.1, 120.0, 0.9, 0.5, 2.0]
        sample_5 = [2.0, 6.0, 0.55, 0.4, 0.2, 50.0, 1.8, 1.0, 0.0]
        sample_6 = [1.0, 3.0, 0.45, 0.5, 0.2, 30.0, 0.7, 1.0, 0.0]
        sample_7 = [4.0, 1.0, 0.5, 0.5, 0.1, 80.0, 1.9, 1.0, 2.0]
        sample_8 = [8.0, 9.0, 0.45, 0.7, 0.15, 220.0, 1.1, 1.0, 1.0]
        samples = [sample_0, sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8]
        return samples

    if sampleId == '10_samples':
        #pitch,roll,x0,y0,subSize,rotAngle,resizeScale,noise, fileNumber
        sample_0 = [4.0, 0.0, 0.65, 0.65, 0.15, 260.0, 0.3, 2.0, 0.0]
        sample_1 = [9.0, 2.0, 0.55, 0.5, 0.15, 320.0, 1.6, 1.5, 1.0]
        sample_2 = [0.0, 4.0, 0.45, 0.4, 0.2, 70.0, 1.1, 1.5, 1.0]
        sample_3 = [3.0, 8.0, 0.4, 0.6, 0.1, 170.0, 0.6, 1.0, 0.0]
        sample_4 = [5.0, 4.0, 0.6, 0.4, 0.1, 240.0, 1.4, 1.0, 1.0]
        sample_5 = [8.0, 10.0, 0.35, 0.65, 0.1, 20.0, 0.8, 1.0, 1.0]
        sample_6 = [2.0, 8.0, 0.7, 0.35, 0.05, 200.0, 1.9, 1.5, 2.0]
        sample_7 = [7.0, 5.0, 0.3, 0.45, 0.15, 80.0, 1.0, 1.0, 2.0]
        sample_8 = [6.0, 7.0, 0.5, 0.3, 0.1, 110.0, 1.7, 0.5, 1.0]
        sample_9 = [9.0, 2.0, 0.55, 0.6, 0.15, 290.0, 1.3, 1.5, 1.0]
        samples = [sample_0, sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8, sample_9]
        return samples

    if sampleId == '5_samples':
        #pitch,roll,x0,y0,subSize,rotAngle,resizeScale,noise, fileNumber
        sample_0 = [5.0, 1.0, 0.35, 0.4, 0.15, 240.0, 0.7, 1.0, 0.0]
        sample_1 = [4.0, 3.0, 0.45, 0.35, 0.2, 30.0, 0.3, 0.5, 2.0]
        sample_2 = [10.0, 7.0, 0.55, 0.55, 0.15, 320.0, 1.1, 1.5, 1.0]
        sample_3 = [8.0, 4.0, 0.65, 0.6, 0.1, 80.0, 1.5, 1.5, 2.0]
        sample_4 = [0.0, 10.0, 0.45, 0.65, 0.1, 210.0, 1.7, 2.0, 0.0]
        samples = [sample_0, sample_1, sample_2, sample_3, sample_4]
        return samples

    if sampleId == 'custom_sample':
        sample_0 = [pitch,roll,x0,y0,subSize,rotAngle,resizeScale,noise,fileNumber]
        samples = [sample_0]
        return samples

if __name__ == '__main__':
    pass
    #generateCentroidPopulation(3)
