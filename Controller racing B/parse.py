import numpy as np 
import math

file = np.loadtxt('traj_race_cl.csv',delimiter=';')

dist_shift = -1.3
arr_new = []
for line in file :
    x,y,theta,vel = line[1], line[2], line[3], line[-2]
    arr_new.append([x+dist_shift*math.cos(theta),y+dist_shift*math.sin(theta),vel])

np.savetxt('lap2.csv', arr_new, delimiter=',')