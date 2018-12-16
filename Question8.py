import random
import numpy as np
import matplotlib.pyplot as plt

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

        expected = compareResult(f(x1), x2)
        perceptRes = g(weight, [1.0, x1, x2])
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
    point1x1 = random.uniform(-1, 1)
    point1x2 = random.uniform(-1, 1)
    point2x1 = random.uniform(-1, 1)
    point2x2 = random.uniform(-1, 1)
    a = abs(point1x1-point2x1)/abs(point1x2-point2x2)
    b = point1x2 - a*point1x1

    def f(x): return a*x+b

    return f
    
#arrays of errors
inSampleErrorSet = []
outSampleErrorSet = []

#Repeating the experiment for 1000 times
for i in range(1000):
    f = target()
    trainingInput = [[1.0, random.uniform(-1,1), random.uniform(-1,1)] for i in range(100)]
    trainingClassification = [compareResult(f(x[1]), x[2]) for x in trainingInput]
    weight = learn(trainingInput, trainingClassification)
    outSampleErrorSet.append(calOutSampleError(weight, f))
    inSampleErrorSet.append(calInSampleError(weight,trainingInput, trainingClassification))

#Calculating and printing the average iteration numbers and average error
inSampleErrorAvg = sum(inSampleErrorSet)/len(inSampleErrorSet)*1.0
outSampleErrorAvg = sum(outSampleErrorSet)/len(outSampleErrorSet)*1.0
print('the average in-sample error is ' + str(inSampleErrorAvg))
print('the average out-sample error is ' + str(outSampleErrorAvg))



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
