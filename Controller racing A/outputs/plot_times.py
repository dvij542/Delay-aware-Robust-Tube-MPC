import math
import numpy as np
import matplotlib.pyplot as plt
import math 

C = 0.21
K = 11
# speeds_time = np.loadtxt('speed_comp.csv')
times = np.loadtxt('../outputs_standalone/comp_times_f.csv')[1:]
# stiffness_time = np.loadtxt('stiff_coeff.csv')
# plt.plot(speeds_time[:,0],speeds_time[:,1])
plt.plot(times[:,0]*1000,label='Predicted computation time')
plt.plot(times[:,1]*1000,label='Actual computation time')
plt.plot(times[:,2]*1000,label='Computation time taken')
# plt.plot(steers_time[:,0],y)
plt.xlabel('Iteration')
plt.ylabel('Computation time (in ms)')
plt.ylim((0,200))
plt.legend()
# plt.plot(np.clip((stiffness_time[:,2]*1.5-stiffness_time[:,1])/(stiffness_time[:,3]*stiffness_time[:,0]),-10,10))
# plt.plot(x+steers_time[0,0],y)
plt.show()