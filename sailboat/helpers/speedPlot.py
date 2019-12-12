import matplotlib.pyplot as plt
import numpy as np
import math
  
def getSpeedForPeriod(windSpeed, regen):
    #speedConversion # y\ =\log_{1.5}.3xx\
    speedConversion = 0
    if (windSpeed < 2):
        speedConversion = windSpeed * 0.87
    elif (windSpeed < 14):
        speedConversion = 0.5 * math.log((windSpeed * 2 * windSpeed), 1.6) - 0.5
    else:
        speedConversion = 5.85
    if regen:
        speedConversion -= 0.25

    return speedConversion

x = []
y = []

for i in np.arange(0, 17, 0.1):
    x.append(i)
    y.append(getSpeedForPeriod(i, False))
  
# plotting the points  
plt.plot(x, y) 
  
# naming the x axis 
plt.xlabel('Wind Speed') 
# naming the y axis 
plt.ylabel('Boat Speed') 
  
# giving a title to my graph 
plt.title('Boat Speed vs Wind Speed') 
  
# function to show the plot 
plt.show() 
