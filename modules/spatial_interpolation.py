import os
import glob
from mat4py import loadmat
import numpy as np
import matplotlib.pyplot as plt

def interpolation(data):
  filename = data['Event']['file_name'] 
  BL = int(filename[1:3]) #getting the bridge length from the file name
  DM = int(filename[9:11])  # getting the damage magnitude (Dead Magnitude) from the file name
  if DM ==0: # if undamaged, it is labeled as DL=0 (DL =  Dead Location)
    DL=0
  else: # if it is damaged, it is labeled as 1(DL=25) and 2 (DL=50)
    DL = int(filename[5:7]) # getting the damage location from the file name
  DM = int(DM/20) #0 (DM=00%), 1 (DM=20%), and 2 (DM=40%)
  DL = int(DL/25) #0 (undamaged), 1 (DL=25%), and 2 (DL=50%)
  a=data['Event']['Sol']['Veh']['A'] # Acceleration data of vehicle both body and axle
  acc = np.array(a)
  v = data['Event']['Veh']['Pos']['vel'] # velocity of the vehicle, required for spatial interpolation
  dt = data['Event']['solver_dt'] # time step for the data collection
  p1 = data['Event']['Veh']['Pos']['t0_ind_beam']-1 # location when it start reaches the bridge.
  p2 = data['Event']['Veh']['Pos']['t_end_ind_beam'] # location when it reaches end of the bridge
  acc = acc[:,p1:p2] # chopping the signal based on the location on the bridge
  # frequency of sampling is 256Hz
  x = v*1/256*np.linspace(p1,p2-1,p2-p1)-100 # obtaining spatial position of the signals
  x_new = np.linspace(0,BL,640) # Equally spaced bridge locations
  tmp = np.zeros((2,640))
  for i in range(2):
    inter_acc[i,:] = np.interp(x_new,x,acc[i,:]) # interpolated signals
  return inter_acc,DM,DL
