'''
Code from http://www.theprojectspot.com/tutorial-post/simulated-annealing-algorithm-for-beginners/6
Translated from java to python
'''

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numba import jit
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

# @jit(nopython=True)
class City:
    x = 0
    y = 0

    def __init__(self, size):
        self.x = random.randint(1, size + 1)
        self.y = random.randint(1, size + 1)
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def distanceTo(self, city):
        xDist = np.abs(self.getX() - city.getX())
        yDist = np.abs(self.getY() - city.getY())
        return np.sqrt( xDist*xDist +  yDist*yDist)

    def __repr__(self):
        return "(" + str(self.getX()) + ", " + str(self.getY()) + ")"

# @jit(nopython=True)
class Map:
    cities = []

    def __init__(self, numCities, size):
        self.cities = []
        for x in range(numCities):
            self.addCity(size)
    
    # def __init__(self):

    def addCities(self, numCities, size):
        for x in range(numCities):
            self.addCity(size)

    def addCity(self, size):
        self.cities.append(City(size))
    
    def printCities(self):
        # for x in self.cities:
        #     print('x: ' + str(x.getX()) + ' ' + str(x.getY()))
        print(self.cities)
    
    def getCity(self, index):
        return self.cities[index]
    
    def numberOfCities(self):
        return len(self.cities)

# @jit(nopython=True)
class Tour:
    tour = []
    distance = 0

    def __init__(self, tour = []):
        self.tour = tour

    def getTour(self):
        return self.tour

    def generateIndividual(self, map):
        self.tour = []
        for cityIndex in range(map.numberOfCities()):
            self.tour.append(map.getCity(cityIndex))
        random.shuffle(self.tour)

    def getCity(self, position):
        return self.tour[position]
    
    def setCity(self, position, city):
        self.tour[position] = city
        self.distance = 0

    def getDistance(self):
        if self.distance == 0:
            tourDistance = 0

            for x in range(self.tourSize()):
                fromCity = self.getCity(x)
                destCity = None
                if x + 1 < self.tourSize():
                    destCity = self.getCity(x + 1)
                else:
                    destCity = self.getCity(0)
                tourDistance += fromCity.distanceTo(destCity)
            self.distance = tourDistance
        return self.distance
    
    def tourSize(self):
        return len(self.tour)
    
    def __repr__(self):
        ret = "|"
        for x in range(self.tourSize()):
            ret += str(self.getCity(x)) + "|"
        return ret

# @jit(nopython=True)
def acceptanceProbability(energy, newEnergy, temperature):
    if newEnergy < energy:
        return 1.0
    return np.exp((float(energy) - float(newEnergy)) / float(temperature))

# @jit(nopython=True)
def plot(tour, size, name):
    x = []
    y = []

    for i in range(tour.tourSize()):
        city = tour.getCity(i)
        x.append(city.getX())
        y.append(city.getY())
    plt.plot(x, y)
    plt.xlabel('East - West') 
    plt.ylabel('North - South')
    plt.title('Map of current best route')
    plt.savefig(name)
    plt.clf()

# @jit(nopython=True)
def trial():
    initTemp = 100000
    temp = initTemp
    coolingRate = 0.99999

    size = 200
    map = Map(10, size)

    currentSolution = Tour()
    currentSolution.generateIndividual(map)
    print(str(currentSolution.getDistance()) + ', ', end='')

    best = currentSolution

    iterations = 0
    plot(best, size, 'before.png')

    # while iterations < 250000:
    while temp > 1:
        iterations += 1
        newSolution = Tour(currentSolution.getTour())

        pos1 = random.randint(0, newSolution.tourSize() - 1)
        pos2 = random.randint(0, newSolution.tourSize() - 1)
        city1 = newSolution.getCity(pos1)
        city2 = newSolution.getCity(pos2)

        newSolution.setCity(pos1, city2)
        newSolution.setCity(pos2, city1)
        
        currentEnergy = currentSolution.getDistance()
        newEnergy = newSolution.getDistance()

        if acceptanceProbability(currentEnergy, newEnergy, temp) > random.random():
            currentSolution = Tour(newSolution.getTour())
        
        if currentSolution.getDistance() < best.getDistance():
            best = Tour(currentSolution.getTour())
        
        temp = temp * coolingRate
        # temp = temp * coolingRate ** (-initTemp / iterations)

        if iterations % 10000 == 0:
            print(str(iterations) + ' ' + str(best.getDistance()) + ' ' + str(temp))
        # temp = -coolingRate*(iterations*iterations) + initTemp

    plot(best, size, 'after.png')
    print(str(best.getDistance()) + ', ', end = '')
    # print(best)
    print(str(iterations) + ', ')


# for x in range(20):
    # print(  str(x) + ', ', end = '')
trial()
