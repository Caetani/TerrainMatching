from commom import *
from generalParameters import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import qmc, hmean, gmean
import pygad
from time import time
from datetime import datetime
from random import shuffle
import concurrent.futures
from memory_profiler import profile

class TerrainMatching:
    def __init__(self):
        self.fileNumber           = None
        self.img                  = None
        self.sub                  = None
        self.resizeScale          = None
        self.x0                   = None
        self.y0                   = None
        self.x1                   = None
        self.y1                   = None
        self.cx                   = None
        self.cy                   = None
        self.rotAngle             = None
        self.pitch                = None
        self.roll                 = None
        self.noiseIntensity       = None
        self.showProgress         = None
        self.numMatches           = None
        self.numFeatures          = None
        self.numBestMatches       = None
        self.numRepetitions       = None
        self.maxNumCores          = None
        self.populationMultiplier = None

    def importMap(self, fileNumber):
        self.fileNumber     = fileNumber
        self.img            = getMap(files[int(fileNumber)], dtype=np.float32)

    def changeSub(self, new_x0, new_x1, new_y0, new_y1, new_rotAngle, pitch, roll, noiseIntensity):
        self.x0, self.y0, self.x1, self.y1 = new_x0, new_y0, new_x1, new_y1
        self.rotAngle, self.pitch, self.roll, self.noiseIntensity = new_rotAngle, pitch, roll, noiseIntensity
        height, width = self.img.shape
        self.x0, self.x1 = round(self.x0*width), round(self.x1*width)
        self.y0, self.y1 = round(self.y0*height), round(self.y1*height)
        #print(f"x0: {self.x0} - x1: {self.x1} - y0: {self.y0} - y1: {self.y1}")
        self.cx, self.cy = round(width/2), round(height/2)
        rotatedPts = rotate2DRectangle(self.x0, self.x1, self.y0, self.y1, cx=self.cx, cy=self.cy, deg=-self.rotAngle)
        for p in rotatedPts:
            x_p, y_p = p
            assert x_p >= 0 and x_p <= width and y_p >= 0 and y_p <= height, "Points out of range."
        imgRot   = rotateMap(self.img, self.rotAngle)
        self.sub = cutSubregion(map=imgRot, x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)
        del imgRot  # To free memory
        self.sub = applyPlaneDistortion(self.sub, pitch, roll)

        #subFotMape = np.copy(self.sub)
        #self.noiseIntensity = 15     # TODO Delete
        

        self.sub = applyNoise(self.sub, self.noiseIntensity, dtype=np.float32)
        #mape = calculate_mape(subFotMape, self.sub)     # TODO Delete
        #rmse = calculate_rmse(subFotMape, self.sub)     # TODO Delete
        #print(f"MAPE: {round(mape, 2)} %\nRMSE: {rmse}")
        return

    def changeImg(self, scale):
        self.resizeScale = scale
        resizedImgHeight, resizedImgWidth = self.img.shape
        resizedImgHeight, resizedImgWidth = round(self.resizeScale*resizedImgHeight), round(self.resizeScale*resizedImgWidth)
        if scale < 1:
            self.img = cv2.resize(self.img, dsize=(resizedImgWidth, resizedImgHeight), interpolation=cv2.INTER_AREA)
        elif scale > 1:
            self.img = cv2.resize(self.img, dsize=(resizedImgWidth, resizedImgHeight), interpolation=cv2.INTER_CUBIC)
        self.x0, self.x1 = self.x0*self.resizeScale, self.x1*self.resizeScale
        self.y0, self.y1 = self.y0*self.resizeScale, self.y1*self.resizeScale
        self.cx = round(resizedImgWidth/2)
        self.cy = round(resizedImgHeight/2)
        return

    def fitnessFunction(self, ga_instance, cromossome, solution_idx):
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

        #printCromossome(cromossome)

        patchSize             = int(patchSize)
        nLevelsSub            = int(nLevelsSub)
        initLevelSub          = int(initLevelSub)
        scaleFactor           = round(scaleFactor, 2)
        fastThreshold         = int(fastThreshold)
        fastBlockSizeInPixels = int(fastBlockSizeInPixels)
        fastFeatureDensity    = round(fastFeatureDensity, 3)
        fastMinThreshold      = int(fastMinThreshold)
        WTA                   = int(WTA)
        crossCheck            = bool(crossCheck)

        edgeThreshold   = patchSize
        matcherType     = getMatcherType(WTA)
        nBlocks         = calculateNumberOfBlocks(lenghtInPixels=fastBlockSizeInPixels, img=self.img)
        fastNumFeatures = calculateNumFastFeatures(fastFeatureDensity, fastBlockSizeInPixels)
        
        windowSizeImg, windowSizeSub = int(windowSizeScaleImg*sigmaImg), int(windowSizeScaleSub*sigmaSub)
        if windowSizeImg % 2 == 0: windowSizeImg = windowSizeImg + 1
        if windowSizeSub % 2 == 0: windowSizeSub = windowSizeSub + 1
        
        inliersHist = []
        scoreHist = []
        histNumBestMatches = []
        #print(f"fitnessFunc 1")
        for i in range(self.numRepetitions):
            localImg = self.img.copy()
            localSub = self.sub.copy()
            localImg = cv2.GaussianBlur(localImg, (windowSizeImg, windowSizeImg), sigmaX=sigmaImg, sigmaY=sigmaImg)
            localSub = cv2.GaussianBlur(localSub, (windowSizeSub, windowSizeSub), sigmaX=sigmaSub, sigmaY=sigmaSub)
            imgGradX = cv2.Sobel(localImg, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            imgGradY = cv2.Sobel(localImg, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
            localImg = cv2.magnitude(imgGradX,imgGradY)
            del imgGradX, imgGradY
            subGradX = cv2.Sobel(localSub, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
            subGradY = cv2.Sobel(localSub, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
            localSub = cv2.magnitude(subGradX,subGradY)
            del subGradX, subGradY
            localImg = float32_to_uint8_GPU(localImg, np.min(localImg), np.max(localImg))
            localSub = float32_to_uint8_GPU(localSub, np.min(localSub), np.max(localSub))
            #print(f"fitnessFunc 2")
            try:
            #for i in range(1):
                orbImg = cv2.ORB.create(nfeatures=self.numFeatures, scaleFactor=scaleFactor, WTA_K=WTA,
                                        edgeThreshold=edgeThreshold, scoreType=cv2.ORB_HARRIS_SCORE,
                                        patchSize=patchSize, fastThreshold=fastThreshold)
                #imgKeypointsFromSegmentation = segmentedFAST_CPU(localImg, fastNumFeatures, nBlocks=nBlocks, minThreshold=fastMinThreshold)
                imgKeypointsFromSegmentation = segmentedFAST_CPU(localImg, fastNumFeatures, nBlocks=nBlocks, minThreshold=fastMinThreshold)
                #print(f"fitnessFunc 3")
                imgKeypoints, imgDescriptor = orbImg.compute(localImg, keypoints=imgKeypointsFromSegmentation)
                #print(f"fitnessFunc 4")
                orbImg = None
                del imgKeypointsFromSegmentation
                orbSub = cv2.ORB.create(nfeatures=self.numFeatures, scaleFactor=scaleFactor, nlevels=nLevelsSub, firstLevel=initLevelSub,
                                        WTA_K=WTA, edgeThreshold=edgeThreshold, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=patchSize,
                                        fastThreshold=fastThreshold)
                subKeypoints, subDescriptor = orbSub.detectAndCompute(localSub, None)
                #print(f"fitnessFunc 5")
                orbSub = None
                del localSub, localImg
                
                bf = cv2.BFMatcher(matcherType, crossCheck=crossCheck)                              
                matches = bf.match(imgDescriptor, subDescriptor)
                #print(f"fitnessFunc 6")
                bf = None
                imgDescriptor, subDescriptor = None, None
                matches = sorted(matches, key=lambda x:x.distance)
                final_matches = matches[:self.numMatches]
                matches = None
                score, inliers, numBestMatches = computeMatchesScore(final_matches, imgKeypoints,
                                                                    subKeypoints, self.resizeScale,
                                                                    self.x0, self.y0, self.cx, self.cy, self.rotAngle,
                                                                    thresholdRadius=5,
                                                                    matcherType=matcherType, pitch=self.pitch, roll=self.roll, numBestMatches=self.numBestMatches)
                #print(f"fitnessFunc 7")
                final_matches = None
                imgKeypoints, subKeypoints = None, None
                inliersHist.append(inliers)
                scoreHist.append(score)
                histNumBestMatches.append(numBestMatches)
                inliers, score, numBestMatches = None, None, None

            except:
                fitness = 10**-9
                if self.showProgress: print(f"\t\tFitness = N/A\t\tInliers = N/A")
                return fitness
            
        inliersInfo = np.mean(inliersHist)
        fitness = np.mean(scoreHist)
        bestMatches = np.mean(histNumBestMatches)
        
        if fitness == 0: fitness = 10**-3
        if self.showProgress: print(f"\t\tFitness = {fitness:_}\tInliers = {inliersInfo} - bestMatches = {bestMatches}")
        return fitness

    def generateInitialPopulation(self, numPop, populationMultiplier=3, numGoodInidividuals=0.25):
        '''
            Function that generates initial population for optimization.

            numPop: desired population size.
            populationMultiplier: establishes the population generated by Latin Hypercube in order to be sorted
                                by their scores.
            numGoodInidividuals: minimum number of individuals with a good score in the initial population.
        '''
        originalNumPop = numPop
        numGoodInidividuals = max(int(numGoodInidividuals*originalNumPop), 1)
        bMinResults = False
        numPop *= populationMultiplier
        print(f"\nGenerating initial population...\n")
        MAX_NUMBER_OF_TRIES = 3
        goodCromossomes = []
        counter = 0
        while not bMinResults and counter < MAX_NUMBER_OF_TRIES:
            print(f"Try {counter+1}/{MAX_NUMBER_OF_TRIES}")
            initialPopulation = generatePopulation(numPop, SPACE_ALL)
            dividedPopulation = np.array_split(initialPopulation, self.maxNumCores)
            with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
                populationScores = []
                fitnessArrays = executor.map(self.evaluatePopulation, dividedPopulation)
                for arr in fitnessArrays:
                    populationScores.append(np.array(arr))
                #populationScores = np.matrix(populationScores, dtype=float)
                #populationScores = np.array(np.swapaxes(populationScores.flatten(), 0, 1))
                populationScores = np.concatenate(populationScores, axis=0).reshape(numPop, 1)
                matrix = np.hstack((initialPopulation, populationScores))
                matrix = matrix[matrix[:, -1].argsort()][::-1]
                if matrix[numGoodInidividuals-1, -1] > 1:
                    bMinResults = True
                else:
                    i = 0
                    while matrix[i, -1] > 1:
                        goodCromossomes.append(matrix[i, :])
                        i += 1
                    counter += 1
        if not bMinResults:
            if len(goodCromossomes) > originalNumPop:
                goodCromossomes = goodCromossomes[goodCromossomes[:, -1].argsort()][::-1]
                result = goodCromossomes
            elif len(goodCromossomes) == originalNumPop:
                result = goodCromossomes
            else:
                if len(goodCromossomes) > 0:
                    goodCromossomes = np.delete(goodCromossomes, [-1], axis=1)
                    popComplement = generatePopulation(originalNumPop-len(goodCromossomes), SPACE_ALL)
                    result = np.vstack((goodCromossomes, popComplement))
                    return result
                else:
                    result = shuffle(initialPopulation)

        print(f"Inial population generated.\n")
        result = np.delete(matrix, [-1], axis=1)
        return result[:originalNumPop]
    
    def evaluatePopulation(self, population):
        fitnessArray = []
        for cromossome in population:
            fitness = self.fitnessFunction(None, cromossome, None)
            fitnessArray.append(fitness)
        assert len(fitnessArray) == len(population), "Error while evaluating population."
        return fitnessArray
    
    def runGenericAlgorithm(self, population_size, num_generations, num_parents_mating, parent_selection_type, keep_elitism, crossover_type, crossover_probability, mutation_type, pct_mutation_num_genes):
        ''' Setting GA hyperparameters '''
        #mutation_probability = 0.2
        parallel_processing = ["process", min(self.maxNumCores, population_size)]
        save_best_solutions = False
        save_solutions = False
        
        initial_population = self.generateInitialPopulation(population_size, populationMultiplier=self.populationMultiplier)
        ga_instance = pygad.GA(num_generations = num_generations, 
                    num_parents_mating = num_parents_mating,
                    gene_space=SPACE_ALL,
                    initial_population = initial_population,
                    fitness_func = self.fitnessFunction,
                    parent_selection_type=parent_selection_type,
                    keep_elitism = keep_elitism,
                    crossover_type = crossover_type,
                    crossover_probability = crossover_probability,
                    mutation_type = mutation_type,
                    #mutation_probability = mutation_probability,
                    mutation_num_genes = mutation_num_genes,
                    save_best_solutions = save_best_solutions,
                    save_solutions = save_solutions,
                    parallel_processing=parallel_processing,
                    on_generation=callback_generation,
                    on_start=callback_generation)
        #ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
        ga_instance.run()
        while ga_instance.run_completed != True:
            time.sleep(0.1)

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"\n\tSolution fitness = {solution_fitness:_}")
        printCromossome(solution)
        print(f"\n\tScore: {solution_fitness}", end = '')
        TM_instance.fitnessFunction(None, solution, None)
        return solution_fitness

def callback_generation(ga_instance):
    print(f"\n\t[Generation: {ga_instance.generations_completed}]")
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    try:
        print(f"\tFitness = {solution_fitness:_}\n\tSolution:", end='')
        printCromossome(solution)
        print()
    except:
        pass

if __name__ == '__main__':
    start_time = time()
    print(f"\nStart of execution...")
    
    TM_instance = TerrainMatching()
    TM_instance.numRepetitions = 1      # No major differences between different numRepetitions.
    TM_instance.numMatches = 100        # TODO Justify a value for numMatches.
    TM_instance.numBestMatches = 10
    TM_instance.numFeatures = 500     # (Sub-region) Limits the amount of computation power demanded.
    TM_instance.showProgress = True    # Used to observe the evolution of the algorithm.
    TM_instance.displayData = False     # Used to observe the behavior of the algorithm on a map.
    TM_instance.populationMultiplier = 2#5   # Number of initial individuals testes before optimization
    TM_instance.maxNumCores = 8

    population_size = 10
    num_generations = 5#10
    numParametersCombinations = 3 #100
    numExecutionsPerHiperparameter = 3 #5
    
    samples = selectSample('9_samples_with_limits')
    samples = [samples[1]]
    GA_parametersPopulation = generatePopulation(numParametersCombinations, GA_PARAMETERS_SPACES)
    assert numParametersCombinations == len(GA_parametersPopulation), f"Error in GA pop gen: {numParametersCombinations} != {len(GA_parametersPopulation)}"

    numMatchesHist, numBestMatchesHist, numFeaturesHist = [], [], []

    sampleNumbersHist = []
    fileNameHist = []
    pitchHist, rollHist = [], []
    x0Hist, y0Hist, x1Hist, y1Hist = [], [], [], []
    subSizeHist, rotAngleHist, imgScaleHist, noiseHist = [], [], [], []

    population_size_hist, num_generations_hist, init_pop_mult_hist          = [], [], []
    num_parents_mating_hist, parent_selection_type_hist, keep_elitism_hist  = [], [], []
    crossover_type_hist, crossover_probability_hist                         = [], []
    mutation_type_hist, mutation_num_genes_hist                             = [], []
    parameterExecution_hist = []

    fitnessHist = []

    for sampleCounter, sample in enumerate(samples):
        print(f"\nSample: {sampleCounter}")
        printSample(sample)

        _pitch, _roll = sample[0], sample[1]
        _x0, _y0     = sample[2], sample[3]
        subSize      = sample[4]
        _rotAngle    = sample[5]
        _imgScale    = sample[6]
        noise        = sample[7]
        fileNumber   = sample[8]
        _x1 = _x0 + subSize
        _y1 = _y0 + subSize

        TM_instance.importMap(fileNumber)
        TM_instance.changeSub(new_x0=_x0, new_x1=_x1, new_y0=_y0, new_y1=_y1, new_rotAngle=_rotAngle, pitch=_pitch, roll=_roll, noiseIntensity=noise)
        TM_instance.changeImg(scale=_imgScale)

        for parameterCounter, GA_parameters in enumerate(GA_parametersPopulation):
            pct_num_parents_mating = GA_parameters[0]
            parent_selection_type  = GA_parameters[1]
            keep_elitism           = GA_parameters[2]
            crossover_type         = GA_parameters[3]
            crossover_probability  = GA_parameters[4]
            #pct_mutation_num_genes = GA_parameters[5]
            pctAdaptiveInit        = GA_parameters[5]
            pctAdaptiveEnd         = GA_parameters[6]

            parent_selection_type = GA_getParentSelectionType(parent_selection_type)
            crossover_type        = GA_getCrossoverType(crossover_type)
            mutation_type         = "adaptive"
            num_parents_mating    = round(pct_num_parents_mating*population_size)
            mutation_num_genes    = (max(round(pctAdaptiveInit*population_size), 1), max(round(pctAdaptiveEnd*pctAdaptiveInit*population_size), 1))
            keep_elitism          = round(keep_elitism)
            crossover_probability = round(crossover_probability, 2)


            for j in range(numExecutionsPerHiperparameter):
                print(f"\n\tSample {sampleCounter+1}/{len(samples)}\n\tParameter {parameterCounter+1}/{len(GA_parametersPopulation)}\n\tExecution {j+1}/{numExecutionsPerHiperparameter}")
                
                #printGAParameters(GA_parameters)

                numMatchesHist.append(TM_instance.numMatches)
                numBestMatchesHist.append(TM_instance.numBestMatches)
                numFeaturesHist.append(TM_instance.numFeatures)

                rollHist.append(TM_instance.roll)
                sampleNumbersHist.append(sampleCounter)
                fileNameHist.append(files[int(fileNumber)])
                pitchHist.append(TM_instance.pitch)
                x0Hist.append(_x0)
                y0Hist.append(_y0)
                x1Hist.append(_x1)
                y1Hist.append(_y1)
                subSizeHist.append(subSize)
                rotAngleHist.append(TM_instance.rotAngle)
                imgScaleHist.append(TM_instance.resizeScale)
                noiseHist.append(TM_instance.noiseIntensity)

                population_size_hist.append(population_size)
                num_generations_hist.append(num_generations)
                init_pop_mult_hist.append(TM_instance.populationMultiplier)        
                num_parents_mating_hist.append(num_parents_mating)
                parent_selection_type_hist.append(parent_selection_type)
                keep_elitism_hist.append(keep_elitism)
                crossover_type_hist.append(crossover_type)
                crossover_probability_hist.append(crossover_probability)                       
                mutation_type_hist.append(mutation_type)
                mutation_num_genes_hist.append(mutation_num_genes)
                parameterExecution_hist.append(j)                       
                #try:
        
                fitness = TM_instance.runGenericAlgorithm(population_size, num_generations, num_parents_mating, parent_selection_type,
                                                          keep_elitism, crossover_type, crossover_probability,
                                                          mutation_type, mutation_num_genes)
                fitnessHist.append(fitness)
                #except: pass

    try:
        end_time = time()
        print(f"\n\n\t\tExecution time: {round(end_time-start_time, 2)} seconds. Saving collected data...")
    except:
        print(f"Exeption in timing measurement.")
    
    pandasDict = {
        "numMatches": numMatchesHist,
        "numBestMatches": numBestMatchesHist,
        "numFeatures": numFeaturesHist,
        "pitch": pitchHist,
        "roll": rollHist,
        "sampleNumbers": sampleNumbersHist,
        "fileName": fileNameHist,
        "x0": x0Hist,
        "y0": y0Hist,
        "x1": x1Hist,
        "y1": y1Hist,
        "subSize": subSizeHist,
        "rotAngle": rotAngleHist,
        "imgScale": imgScaleHist,
        "noise": noiseHist,
        "population_size": population_size_hist,
        "num_generations": num_generations_hist,
        "init_pop_mult": init_pop_mult_hist,    
        "num_parents_mating": num_parents_mating_hist,
        "parent_selection_type": parent_selection_type_hist,
        "keep_elitism": keep_elitism_hist,
        "crossover_type": crossover_type_hist,
        "crossover_probability": crossover_probability_hist,                     
        "mutation_type": mutation_type_hist,
        "mutation_num_genes": mutation_num_genes_hist,
        "parameterExecutionCount": parameterExecution_hist,            
        "fitness": fitnessHist
    }
    df = pd.DataFrame(pandasDict)
    try:
        now = datetime.now()
        time = now.strftime("%Hh_%Mm_date_%d_%m_%Y")
        df.to_csv(f"GA_Parameters_Result_{time}.csv")
    except:
        print(f"Exeption in CSV generation.")
        df.to_csv(f"GA_Parameters_Result.csv")

    print(f"Data saved in CSV file. End of execution.")
