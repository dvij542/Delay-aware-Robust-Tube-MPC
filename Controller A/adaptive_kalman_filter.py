# from mpc_robust import FIRST_TIME
import numpy as np
import matplotlib.pyplot as plt
import math

FIRST_TIME = True
w_bar = 0
e_bar = 0
x_hat = 0
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
Theta = np.array([[1], 
                  [0]])
F = np.array([[1, 0], 
              [0, 1]])
Nt = 30
forgetting_factor = (Nt-1)/Nt

def update_delay_time_estimate(z) :
    global Theta,x_hat,P,Q,e_bar,R,w_bar,F
    # print(Theta[0,0], Theta[1,0])
    if FIRST_TIME :
        x_hat = z
        return x_hat,0
    x_hat_bar = Theta[0,0]*x_hat + Theta[1,0]
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
    return Theta[0,0]*x_hat + Theta[1,0], math.sqrt(P)

