import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.patches as patches


times = np.loadtxt('computation_times.csv')
plt.plot(times[:,3], times[:,0], label="Predicted delay time")
plt.plot(times[:,3], times[:,1], label="Actual delay time")
plt.plot(times[:,3], times[:,2], label="Predicted upper bound on delay time")
plt.ylim([0,0.15])
# plt.title("Predicted vs actual computation time")
plt.xlabel("ROS time")
plt.ylabel("Delay time")
plt.legend()
plt.savefig("computation_time.png", format='png')
plt.show()
