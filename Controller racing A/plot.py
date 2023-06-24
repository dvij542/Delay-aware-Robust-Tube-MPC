import numpy as np
import matplotlib.pyplot as plt
import math

file0 = np.loadtxt("front_right_wheel_0.5.csv")

times = file0[:30,0]*10 - file0[0,0]*10
responses = -file0[:30,1]
k1 = 30
c1 = 20.2
responses_expected = c1*(1-np.exp(-k1*times))
# plt.plot(times,responses,'-',label="observed response for delta=0.5")
# plt.plot(times,responses_expected,'--', label="expected response for delta=0.5,k=3")
# plt.show()

file0 = np.loadtxt("front_right_wheel_-1.csv")

times = file0[:30,0] - file0[0,0]
responses = -file0[1:31,1]
k2 = 30
c2 = 43.4
responses_expected = c2*(1-np.exp(-k2*times))
plt.plot(times,responses,'-',label="Observed unit step response")
plt.plot(times,responses_expected,'--',label="Estimated unit step response with K=30 hz")
plt.xlabel("Time")
plt.ylabel("Front right wheel yaw angle")
# plt.title("Steering angle response")
plt.legend()
plt.savefig("responses.png")
plt.show()