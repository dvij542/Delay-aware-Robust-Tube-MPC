import math
import numpy as np
import matplotlib.pyplot as plt
import math 

C = 6.3
K = 0.09
speeds_time = np.loadtxt('speed_comp.csv')[4:-30]
print(speeds_time)
# steers_time = np.loadtxt('steer_const.csv')[4:]
# stiffness_time = np.loadtxt('stiff_coeff.csv')
x = np.array(speeds_time[:,1])
# print(x)
y = C - K*x
plt.plot(speeds_time[:,0],speeds_time[:,1],label='Speed on full throttle')
# plt.plot(steers_time[:,0],(math.pi/180)*steers_time[:,1]*3/steers_time[:,2],label='Wheel angle from simple kinematic model')
# plt.plot(steers_time[:,0],y,label='Wheel angle from equation (K=11)')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Speed')
# plt.plot(np.clip((stiffness_time[:,2]*1.5-stiffness_time[:,1])/(stiffness_time[:,3]*stiffness_time[:,0]),-10,10))
# plt.plot(x,y)
plt.show()