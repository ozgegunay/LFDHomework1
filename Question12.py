import random
import numpy as np
import matplotlib.pyplot as plt
import operator

#hypothesis function
def g(weight, x):
    return np.sign(np.dot(x, weight))

#learning algorithm
def learn(trainingSet, trainingClassification):
    return np.matmul(np.linalg.pinv(trainingSet), trainingClassification)

#Calculating in-sample error
def calInSampleError(weight, trainingSet, trainingClassification):
    error = 0
    for i in range(len(trainingSet)):
        res = g(weight, trainingSet[i])
        if(res != trainingClassification[i]):
            error += 1
    return error/len(trainingSet)*1.0

#Calculating out sample error
def calOutSampleError(weight, f):
    error = 0
    for i in range(10000):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)

        expected = np.sign(f(x1, x2))
        perceptRes = g(weight, [1.0, x1, x2, x1*x2, x1**2, x2**2])
        if(expected != perceptRes):
            error += 1
    return error/10000.0        

#Compares the classification with the label
def compareResult(y,expectedY):
    if(y > expectedY):
        return 1
    elif(y == expectedY):
        return 0
    else:
        return -1

#Creating target function
def target():
    def f(x1, x2): return x1**2 + x2**2 - 0.6
    return f

#Finds most common weight value in the given weight column
def findMostCommon(weight, index):
    weightDict = {}
    for i in range(len(weight)):
        w = weightDict.get(weight[i][index],-1)
        if(w == -1):
            weightDict[weight[i][index]] = 1
        else:
            weightDict[weight[i][index]] += 1
    return max(weightDict.items(), key=operator.itemgetter(1))[0]

#Creating feature vector which is (1,x1,x2,x1*x2, x1**2,x2**2)
def createFeatureVector(N):
    featureVector = []
    for a in range(N):
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        featureVector.append([1.0, x1, x2, x1*x2, x1**2, x2**2])
    return featureVector

#arrays of errors and weights of 1000 experiment
inSampleErrorSet = []
outSampleErrorSet = []
weightSet = []

#Repeating the experiment for 1000 times
for i in range(1000):
    weight = [0, 0, 0, 0, 0, 0]
    f = target()
    trainingInput = createFeatureVector(1000)
    trainingClassification = [np.sign(f(x[1], x[2])) for x in trainingInput]
    for y in range(100):
        rIndex = int(random.uniform(0,1000))
        trainingClassification[rIndex] *= -1
    weight = learn(trainingInput, trainingClassification)
    weightSet.append(weight)
    outSampleErrorSet.append(calOutSampleError(weight, f))
    inSampleErrorSet.append(calInSampleError(weight,trainingInput, trainingClassification))

#Finds the most common values for weight columns
commonWeight = [findMostCommon(weightSet, 0), findMostCommon(weightSet, 1), findMostCommon(weightSet, 2), 
findMostCommon(weightSet, 3), findMostCommon(weightSet, 4), findMostCommon(weightSet, 5)]

#Calculating and printing the average iteration numbers and average error
inSampleErrorAvg = sum(inSampleErrorSet)/len(inSampleErrorSet)*1.0
outSampleErrorAvg = sum(outSampleErrorSet)/len(outSampleErrorSet)*1.0
print('the average in-sample error is ' + str(inSampleErrorAvg))
print('the average out-sample error is ' + str(outSampleErrorAvg))
print('common weight vector is ' + str(commonWeight)) 



# # This is target (f(x)) function
# slope2, intercept = np.polyfit([point1x1, point2x1], [point1x2, point2x2], 1)
# plt.plot([point1x1, point2x1], [point1x2, point2x2], 'ro', color='y')
# plt.plot(np.arange(-1, 1, 0.01),
#          [slope2*i + intercept for i in np.arange(-1, 1, 0.01)], 'b')
# plt.plot([x[0] for x, y in zip(chosenInputs, chosenValues) if y > 0], [x[1]
#                                                                        for x, y in zip(chosenInputs, chosenValues) if y > 0], 'ro', color='c')
# plt.plot([x[0] for x, y in zip(chosenInputs, chosenValues) if y < 0], [x[1]
#                                                                        for x, y in zip(chosenInputs, chosenValues) if y < 0], 'ro', color='m')
# plt.plot([x[0] for x, y in zip(chosenInputs, chosenValues) if y == 0], [x[1]
#                                                                         for x, y in zip(chosenInputs, chosenValues) if y == 0], 'ro', color='w')
# plt.title('f(x)')
# plt.show()
# plt.plot(np.arange(-1, 1, 0.01),
#          [slope2*i + intercept for i in np.arange(-1, 1, 0.01)], 'b')
# plt.show()
