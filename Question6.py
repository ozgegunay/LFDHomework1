import random
import numpy as np
import matplotlib.pyplot as plt

#hypothesis function
def g(weight, x):
    return np.sign(np.dot(x, weight))

#learning algorithm
def learn(weight, trainingSet, trainingClassification):
    iteration=0
    while True:
        iteration += 1
        misses = findMissclassified(trainingSet, trainingClassification, weight)
        if(len(misses) == 0):
            break
        randomIndex = int(random.uniform(0, len(misses)))
        missIndex = misses[randomIndex]
        weight = np.add(weight, [x*trainingClassification[missIndex] for x in trainingSet[missIndex]])
    return weight, iteration

#Finds missclassified points 
def findMissclassified(trainingSet, traningClassification, weight):
    missClassified = []
    for i in range(len(trainingSet)):
        result = g(weight, trainingSet[i])
        if(result != trainingClassification[i]):
            missClassified.append(i)
    return missClassified

#calculates error
def calError(weight, f):
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

#arrays of errors and iteration numbers  
errorset = []
iterationSet = []

#Repeating the experiment for 1000 times
for i in range(1000):
    weight = [0, 0, 0]
    f = target()
    trainingInput = [[1.0, random.uniform(-1,1), random.uniform(-1,1)] for i in range(100)]
    trainingClassification = [compareResult(f(x[1]), x[2]) for x in trainingInput]
    weight, iteration = learn(weight, trainingInput, trainingClassification)
    iterationSet.append(iteration)
    errorset.append(calError(weight, f))

#Printing the iteration numbers and average error
iterAvg = sum(iterationSet)/len(iterationSet)*1.0
errorAvg = sum(errorset)/len(errorset)*1.0
print('the average iteration number is ' + str(iterAvg))
print('the average error is ' + str(errorAvg))



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
