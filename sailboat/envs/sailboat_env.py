import gym
from gym import error, spaces, utils
from gym.utils import seeding

import csv
import os
import random

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import math

'''
1852 meters/nautical mile
distance will be tracked in meters

reward will be based on the number of meters traveled in a period

if battery is at 0 there will be a negative reward no matter what

Things that need to be controlled:
regen (yes/no)
store h2 (%)?
draw h2
run motor (yes/no) - can't do at the same time as regen default if both yes run motor

Description:
  A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
Source:
  This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

Observation: 
  Type: Box(4)
  Num	Observation                 Min         Max
 -1 Period Rep                    0           100
 -2 Period Rep                    0           100
  3 Battery Percentage            0           100
  4 H2 Percentage                 0           100
  5 Solar kw/m^2 D_0              0           40
  6 Solar kw/m^2 D_1              0           40
  7 Solar kw/m^2 D_2              0           40
  8 Solar kw/m^2 D_3              0           40
  9 Solar kw/m^2 D_4              0           40
  10 Solar kw/m^2 D_5              0           40
  11 Wind m/s D_0                  0           ~15
  12 Wind m/s D_1                  0           ~15
  13 Wind m/s D_2                  0           ~15
  14 Wind m/s D_3                  0           ~15
  15 Wind m/s D_4                  0           ~15
  16 Wind m/s D_5                  0           ~15

      
Actions:
  Type: Discrete(2)
  Num	Action
  0 Run Regen (On, off)
  1 Run Motor (0-1) (0% - 100% of capability)
  2 Store H2 (0-1) (0% - 100% of excess power)
  3 Draw H2 (0-1) (0% - 100% of capabiliry)
      
  
Reward:
  One point for every 100 meters traveled. 0 if battery is empty.

Starting State:
  

Episode Termination:
  Battery is dead for more than 2 days (6 periods)
  2 Days with little to no movement (not sure here)
  Boat reaches goal

  '''

def close_event():
    plt.close()

PERIOD_REP = [
  [0.87, 0.5],
  [0, 1],
  [-0.87, 0.5],
  [-0.87, -0.5],
  [0, -1],
  [0.87, -0.5]
]

PERIOD_GEN = [0, 0.02, 0.38, 0.42, 0.08, 0]

PERIOD_DRAW = [0.696, 0.696, 0.6, 0.6, 0.6, 0.696]

DIRNAME = os.path.dirname(__file__)
FILENAME = os.path.join(DIRNAME, 'data.csv')

def moving_average(a, n=15) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

class Sailboat(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.data = []
    self.trial_count = 0
    self.getData()
    self.battery = 0 # 100% of capacity
    self.battery_capacity = 20
    self.h2 = 20 # 100% of capacity
    self.distance = 0 # Distance Traveled
    self.goal = 2700 * 1852 # nautical miles Canary Islands to Windward islands
    self.day = 0
    self.period = 2 # (0-6)
    self.periodCount = 0
    self.reward = 0
    self.data_index = random.randint(0, len(self.data) - 1)
  
    self.solarTrack = []
    self.speedTrack = []
    self.distanceTrack = []
    self.periodTrack = []
    self.dayTrack = []
    self.periodCountTrack = []
    self.batteryTrack = []
    # Example when using discrete actions:
    # self.action_space = spaces.Discrete(2)
    self.action_space = spaces.Box(low=0, high=1, shape=(2,1), dtype=np.float32) 
    # # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=100, shape=(14, 1), dtype=np.float32)

  def reset(self):
    self.trial_count += 1
    self.battery = 0 # 100% of capacity
    self.h2 = 100 # 100% of capacity
    self.distance = 0 # Distance Traveled
    self.goal = 2700 * 1852 # nautical miles Canary Islands to Windward islands
    self.day = 0
    self.period = 2 # (0-6)
    self.periodCount = 0 
    self.reward = 0
    self.data_index = random.randint(0, len(self.data) - 1)
    self.speedTrack = []
    self.solarTrack = []
    self.distanceTrack = []
    self.periodTrack = []
    self.dayTrack = []
    self.periodCountTrack = []
    self.batteryTrack = []
    return self.getObservation()

  def getData(self):
    with open(FILENAME, newline='') as csvfile:

      reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      headerFound = False
      for row in reader:
          # print(': '.join(row))
          if (not headerFound):
              if (row[0] == 'ENDHEADER'):
                  headerFound = True
          else :
              self.data.append([float(row[2]), float(row[3])])
  
  def getPowerForPeriod(self):
    solarEff = 0.2 # Optomistic pv efficency
    return self.data[self.data_index][1] * solarEff * PERIOD_GEN[self.period]
  

  def getSpeedForPeriod(self, regen):
      #speedConversion # y\ =\log_{1.5}.3xx\
      speedConversion = 0
      # windSpeed = self.data[self.data_index][0]
      windSpeed = 6
      if (windSpeed < 2):
          speedConversion = windSpeed * 0.87
      elif (windSpeed < 14):
          speedConversion = 0.5 * math.log((windSpeed * 2 * windSpeed), 1.6) - 0.5
      else:
          speedConversion = 5.85

      if regen:
          speedConversion -= 0.25

      return speedConversion


  def getNextIndex(self, index):
    # print(len(self.data))
    # print(self.index)
    if (index == len(self.data) - 1):
      return 0
    return index + 1
  
  def getComingDays(self):
    index = self.data_index
    comingDays = []
    for x in range(5):
      comingDays.append(float(self.data[index][0]))
      comingDays.append(float(self.data[index][1]))
      index = self.getNextIndex(index)
    return comingDays

  def getObservation(self):
    ret = []
    ret.append(PERIOD_REP[self.period][0])
    ret.append(PERIOD_REP[self.period][1])
    ret.append(self.battery)
    ret.append(self.h2)
    ret += self.getComingDays()
    return ret

  def setNextPeriod(self):
    self.period = 1 + self.period
    if self.period == 6:
      self.data_index = self.getNextIndex(self.data_index)
      self.day += 1
      self.period = 0

  def getSolarForPeriod(self):
    solar = self.data[self.data_index][1]
    index = PERIOD_GEN[self.period]
    solarEfficency = 0.19 # average ifficency of a solar panel
    solarArea = 13 # m^2 of solar
    return solar * index * solarEfficency * solarArea

  def isDone(self):
    if self.goal < self.distance:
      return True
    else:
      return False
# Actions:
#   Type: Discrete(2)
#   Num	Action
#   0 Run Regen (On, off)
#   1 Run Motor (0-1) (0% - 100% of capability)
#   2 Store H2 (0-1) (0% - 100% of excess power)
#   3 Draw H2 (0-1) (0% - 100% of capabiliry)

  def step(self, actions):
    # self.distance += self.getSpeedForPeriod()
    # print(actions)
    powerGen = 0

    powerGen += self.getSolarForPeriod()
    self.solarTrack.append(powerGen)

    if actions > 2:
      speed = self.getSpeedForPeriod(True)
      powerGen += speed/10 * 4
    else:
      speed = self.getSpeedForPeriod(False)

    self.battery += powerGen

    draw = PERIOD_DRAW[self.period]

    self.battery -= draw
    # print('Power Gen: ' + str(powerGen))
    # print('Battery: ' + str(self.battery))
    self.batteryTrack.append(self.battery)
    # print(actions)
    if actions >= 1:
      maxdraw = self.battery
      powerToMotor = actions * 2000
      if powerToMotor >= maxdraw:
        powerToMotor = maxdraw - 1
      self.battery -= powerToMotor
      # speedFromMotor = powerToMotor * 0.0003
      speedFromMotor = powerToMotor * 0.1
      speed += speedFromMotor

    self.periodCount += 1
    self.distance += speed * 60 * 60 * 4 # m/s * 60min * 60hr * 4 hrs
    self.speedTrack.append(speed)
    self.distanceTrack.append(self.distance)
    self.periodTrack.append(self.period)
    self.periodCountTrack.append(self.periodCount)
    self.dayTrack.append(self.day)
    # print('powergen: ' + str(powerGen))

    self.setNextPeriod()

    # if (self.isDone()):
    # with open('results.csv', 'w', newline='') as csvfile:
    #   res_writer = csv.writer(csvfile, delimiter=',',
    #     quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #   res_writer.writerow([self.periodCount, actions[0], actions[1]])
    # print(self.trial_count)
    # return self.getObservation(), speed, self.isDone(), {}
    reward = speed
    if self.battery < 10:
      a = self.battery / 20
      reward = speed * a
    
    if self.battery > 50:
      reward -= self.battery * 2

    return self.getObservation(), reward, self.isDone(), {}
    # return observation, reward, done, info

  def render(self, e, history, truth):
     
    # Data
    if truth :
      df=pd.DataFrame({'x': self.periodCountTrack,
      'Distance 1000km': (np.array(self.distanceTrack) / 1000000),
      'Speed': (np.array(self.speedTrack)), 
      'Solar': (np.array(self.solarTrack)),
      'Battery': (np.array(self.batteryTrack))
      })
      df2=pd.DataFrame({'x': len(history),
      'History': history
      })
      # if(self.isDone()):
      if(self.isDone()):
        # print("Day: " + str(self.day))
        # print("Period: " + str(self.periodCount))
        fig, p1 = plt.subplots(3, 1)
        p1[0].plot('x', 'Distance 1000km', data=df) 
        p1[0].plot('x', 'Speed', data=df)
        p1[0].plot('x', 'Solar', data=df)
        p1[0].set_xlabel('Period of Journey') 
        p1[0].set_ylabel('Dist/Power (m/KWh)')
        p1[0].axis([0, 100, 0, 8])
        p1[0].legend()

        p1[1].plot('x', 'Battery', data=df)
        p1[1].set_xlabel('Period of Journey') 
        p1[1].set_ylabel('Power (KWh)') 
        p1[1].axis([0, 100, 0, 50])
        
        p1[2].plot(moving_average(np.asarray(history)))
        p1[2].set_xlabel('Epoch')
        p1[2].set_ylabel('Score (lower is better)')

        # p1[2].title('Trining Data (lower is better)')

        # fig.tight_layout()
        # timer = fig.canvas.new_timer(interval = 2000)
        # timer.add_callback(close_event)
        # timer.start()
        plt.savefig('./results/' + str(e).zfill(3) + '-result.png')
    else:
      plt.plot(moving_average(np.asarray(history)))
      plt.xlabel('Epoch')
      plt.ylabel('Score (lower is better)')
      plt.savefig('./results/history.png')
      # plt.show()
    plt.close()

# Save to a video with
# ffmpeg -framerate 1 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4

# def close(self):
#   ...