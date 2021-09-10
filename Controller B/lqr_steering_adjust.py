from casadi import *

import numpy as np
import math

T = .04
N = 4

K = (1-math.e**(-30*T/2))
theta_ac = SX.sym('theta_ac')
theta = SX.sym('theta')

n_states = 1
n_controls = 1

U=SX.sym('U',N)
P=SX.sym('P',n_states + N)
X=SX.sym('X',N)
X[0] = P[0]
X[1] = P[0] + K*(U[0]-P[0])

for k in range(1,N-1,1) :
    X[k+1] = X[k] + K*(U[k]-X[k]) + K*(U[k+1]-(X[k] + K*(U[k]-X[k])))

obj=0
Q=SX(5)
R=SX(0)
R2=SX(0)

for k in range(N) :
    obj = obj + Q*(P[k+1]-X[k])**2 + R*U[k]**2

for k in range(N-1) :
    obj = obj + R2*(U[k+1] - U[k])**2


nlp_prob_B = {'f': obj, 'x': U, 'p': P}
options_B = {
            'ipopt.print_level' : 0,
            'ipopt.max_iter' : 150,
            'ipopt.mu_init' : 0.01,
            'ipopt.tol' : 1e-8,
            'ipopt.warm_start_init_point' : 'yes',
            'ipopt.warm_start_bound_push' : 1e-9,
            'ipopt.warm_start_bound_frac' : 1e-9,
            'ipopt.warm_start_slack_bound_frac' : 1e-9,
            'ipopt.warm_start_slack_bound_push' : 1e-9,
            'ipopt.mu_strategy' : 'adaptive',
            'print_time' : False,
            'verbose' : False,
            'expand' : True
        }

solverB=nlpsol("solver","ipopt",nlp_prob_B,options_B)
lbx_B = np.zeros(N)
ubx_B = np.zeros(N)

for k in range (0,N,1): 
    lbx_B[k]=-math.pi/4.5
    ubx_B[k]=math.pi/4.5

x0=np.random.rand(N)

def get_optimal_commands(comm_seq, curr_command) :
    p = [curr_command]
    for i in range(N) :
        p.append(float(comm_seq[i]))
    
    sB = solverB(x0=x0,p=p,lbx=lbx_B,ubx=ubx_B)
    x = sB['x']
    return x
