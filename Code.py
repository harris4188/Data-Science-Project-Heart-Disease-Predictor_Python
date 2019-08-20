import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import expit

a = []
b = []
data = []
SaveData = []

with open('data.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        Array = np.array(row)
        ColumnCount = Array.shape
        break
         
data = np.empty((0, ColumnCount[0]))
SaveData = np.empty((0, ColumnCount[0]+1))

with open('data.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        a.append(row[0])
        b.append(row[1])
         
        array = np.array(row)
        array.shape = (1, ColumnCount[0])
        data = np.concatenate((data, array))
         

listofOnes = [1] * data.shape[0]
listofOnes = np.array(listofOnes)

data = data.astype(np.float)
num = data[0].size
print(num)

Matrix = data[:, 0:num]
listofOnes.shape = (data.shape[0], 1)
x = np.concatenate((listofOnes, Matrix), axis=1)

print(x[0])
print(x[1])
print(x.shape)

# y=data[:,num-1]
y = []
with open('target.csv', 'rt') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        y.append(row[0])

y = np.array(y)
y = y.astype(np.float)

print(y)

theeta = [0] * x.shape[1]
theeta = np.array(theeta)
theeta.shape = (x.shape[1], 1)
# theeta=np.transpose(theeta)
theeta = theeta.astype(np.float)


temptheeta = [0] * x.shape[1]
temptheeta = np.array(temptheeta)
temptheeta.shape = (x.shape[1], 1)
temptheeta = temptheeta.astype(np.float)


theeta[0] = 0
theeta[1] = 0
theeta[2] = 0

print(theeta)
print(theeta.shape)
print(temptheeta)
print(temptheeta.shape)

std = np.std(x, axis=0)
mean = np.mean(x, axis=0)

print('**************************')
print(std)
print(mean)

for i in range(1, num+1):
    for j in range(0, data.shape[0]):
        x[j][i] = x[j][i]-mean[i]
        x[j][i] = x[j][i]/std[i]

sum = 0
learning_rate = 2.0
iteration = 50


def hyp(lists):
    # print(lists)
    result = np.dot(lists, theeta)
    result = 1+math.e**(-1*result)
    result = 1/result
    return result


def calculateSumforCostfunction():
    sum=0
    for index in range(0, data.shape[0]):
        sum=sum+(y[index]*math.log(hyp(x[index]))+(1-y[index])*math.log(1-hyp(x[index])))
        # print (sum)
    return sum    


def calculateSumtoupdateTheeta(xValue):
    sum = 0
    for index in range(0, data.shape[0]):      
        sum = sum+(hyp(x[index])-y[index])*(x[index][xValue])
    # print("sum ",sum)
    return sum    

# Gradient Descent to find theetas

costs = []

for i in range(0, iteration):
    addValue = []
    # print(theeta)

    sum = calculateSumforCostfunction()
    # print (sum)
    cost = -1*sum/(data.shape[0])
    # print (cost)
    costs.append(cost)

    addValue.append(cost)
    for k in range(0, x.shape[1]):
        addValue.append(theeta[k])

    for j in range(0, x.shape[1]):
        sum = calculateSumtoupdateTheeta(j)
        # print(theeta[j]-learning_rate*(sum/data.shape[0]))
        temptheeta[j] = theeta[j]-learning_rate*(sum/data.shape[0])

    for n in range(0, x.shape[1]):
        theeta[n] = temptheeta[n]

numberOfIterations = np.arange(50)
print(costs)

plt.xlabel('Number of Iterations')
plt.ylabel('Cost J')
plt.plot(numberOfIterations, costs, 'green', label='alpha = 0.01')
plt.show()

# Testing

test = open('test.csv').read().splitlines()

for x in range(len(test)):
    test[x] = test[x].split(',')
    for y in range(len(test[x])):
        test[x][y] = float(test[x][y])

print('-----------------------')
print(test)

print(len(test[0]))

for x in range(len(test)):
    for y in range(len(test[x])):
        test[x][y] = float((test[x][y]-mean[y+1]))/float(std[y+1])

test = np.array(test)

x0 = np.ones(len(test))
print(x0)

x0 = x0.reshape(21, 1)

print(x0.shape)
print(test.shape)

print(x0)

test = np.concatenate((x0, test), axis=1)

print(test)

result = hyp(test)
print(result)

for x in range(len(result)):
    if result[x] < .5:
        result[x] = 0
    elif result[x] < 1:
        result[x] = 1

print(result)
