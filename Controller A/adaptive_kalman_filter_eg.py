import numpy as np
import matplotlib.pyplot as plt
import math

times = np.loadtxt('computation_times1.csv')

w_bar = 0
e_bar = 0
x_hat = times[0,1]
# print(x_hat)
P = 0
Nr = 30
Nq = 30
Q = 0.00001
R = 0.00001
A = 1
alpha_1 = (Nr-1)/Nr
alpha_2 = (Nq-1)/Nq
i = 1
beta = 5
times[0,0] = times[0,1]
times[0,2] = times[0,1]
Theta = np.array([[1], 
                  [0]])
F = np.array([[1, 0], 
              [0, 1]])
Nt = 30
forgetting_factor = (Nt-1)/Nt

for z in times[1:,1] :
    print(Theta[0,0], Theta[1,0])
    x_hat_bar = Theta[0,0]*x_hat + Theta[1,0]
    times[i,0] = x_hat_bar
    times[i,2] = x_hat_bar + beta*math.sqrt(P)
    A = Theta[0,0]
    P_bar = A*P*A + Q
    e = z - x_hat_bar
    e_bar = alpha_1*e_bar + (1-alpha_1)*e
    delta_R = (1/(Nr-1))*(e-e_bar)**2 - (1/Nr)*P_bar
    R = math.fabs(alpha_1*R + delta_R)
    K = P_bar/(P_bar+R)
    x_hat = x_hat_bar + K*e
    P_prev = P
    P = (1-K)*P_bar
    w_hat = x_hat - x_hat_bar
    w_bar = alpha_2*w_bar + (1-alpha_2)*w_hat
    delta_Q = (1-alpha_2)*(P-P_prev) + (1/(Nq-1))*(w_hat-w_bar)**2
    Q = math.fabs(alpha_2*Q + delta_Q)
    phi = np.array([[x_hat],
                    [1]])
    F = F - np.matmul(np.matmul(np.matmul(F,phi),phi.T),F)/(forgetting_factor+np.matmul(np.matmul(phi.T,F),phi))
    F/=forgetting_factor
    Theta = Theta + np.matmul(F,phi)*(x_hat-x_hat_bar)
    i+=1


# times_dash = times.copy()
# Q = 0.00004
# R = 0.00004
# times_dash[0,0] = times_dash[0,1]
# times_dash[0,2] = times_dash[0,1]
# i = 1
# P = 0

# for z in times[1:,1] :
#     x_hat_bar = A*x_hat
#     # print(P)
#     times_dash[i,0] = x_hat_bar
#     times_dash[i,2] = x_hat_bar + beta*math.sqrt(P)
    
#     P_bar = A*P*A + Q
#     e = z - x_hat_bar
#     e_bar = alpha_1*e_bar + (1-alpha_1)*e
#     delta_R = (1/(Nr-1))*(e-e_bar)**2 - (1/Nr)*P_bar
#     R = math.fabs(alpha_1*R + delta_R)
#     K = P_bar/(P_bar+R)
#     x_hat = x_hat_bar + K*e
#     P_prev = P
#     P = (1-K)*P_bar
#     w_hat = x_hat - x_hat_bar
#     w_bar = alpha_2*w_bar + (1-alpha_2)*w_hat
#     delta_Q = (1-alpha_2)*(P-P_prev) + (1/(Nq-1))*(w_hat-w_bar)**2
#     Q = math.fabs(alpha_2*Q + delta_Q)
#     i+=1

print("Completed")
plt.plot((times[:,3]-times[0,3]), times[:,0]*1000, label="Predicted computation time")
plt.plot((times[:,3]-times[0,3]), times[:,1]*1000, label="Actual computation time")
plt.plot((times[:,3]-times[0,3]), times[:,2]*1000, label="Computation time taken")
# plt.plot(times_dash[:,3], times_dash[:,0], label="Predicted computation time with Q = 0.0004")
# plt.plot(times_dash[:,3], times_dash[:,2], label="Computation time taken with R = 0.0004")

plt.ylim([0,100])
# plt.title("Predicted vs actual computation time")
plt.xlabel("ROS time (in s)")
plt.ylabel("Computation time (in ms)")
plt.legend()
plt.savefig("computation_time_final.png", format='png')
plt.show()


