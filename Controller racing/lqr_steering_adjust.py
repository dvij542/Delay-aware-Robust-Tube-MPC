from casadi import *

import numpy as np
import math

T = .01
N = 10
safety_dist = 8
K = (1-math.e**(-30*T))/T
gamma = 2

x=SX.sym('x')
y=SX.sym('y')
theta=SX.sym('theta')
pedal=SX.sym('pedal')
delta_ac=SX.sym('delta_ac')
v=SX.sym('v')
states=vertcat(x,y,theta,v,delta_ac)
delta=SX.sym('delta')
controls=vertcat(pedal,delta)
EPSILON = 0
Ka = 10
Kf = -0.25
###########    Model   #####################
rhs=[
        v*cos(theta+EPSILON),
        v*sin(theta+EPSILON),
        v*tan(delta_ac+K*(delta-delta_ac)*T/2)/2.9,
        Ka*pedal+Kf,
        K*(delta-delta_ac)
    ]                                            
rhs=vertcat(*rhs)
f=Function('f',[states,controls],[rhs])

n_states=5
n_controls=2
U=SX.sym('U',n_controls+1,N)
g=SX.sym('g',N)
viol_amt = SX.sym('V',N)
viol_amt2 = SX.sym('VV',N)
# P=SX.sym('P',1 + 2*N + 3 + 5)
X=SX.sym('X',n_states,(N+1))
# X[:,0]=P[0:1]         
P=SX.sym('P',1 + N + 5)
X[:3,0] = 0
X[3,0] = P[-1]
X[4,0] = P[0]
# X[1] = P[0] + K*(U[0]-P[0])
for k in range(0,N,1):
        st=X[:,k]
        con=U[:-1,k]
        f_value=f(st,con)
        st_next=st+(T*f_value)
        X[:,k+1]=st_next

obj=0
Q=SX(5)
R=SX(0)
R2=SX(1000)
R_viol = 10
x_obs = P[-5]
y_obs = P[-4]
vx_obs = P[-3]
vy_obs = P[-2]

for k in range(N) :
    x_obs_curr = x_obs + vx_obs*k*T
    y_obs_curr = y_obs + vy_obs*k*T
    # viol_amt[k] = g[k]-((X[0,k]-x_obs_curr)**2 + (X[0,k]-x_obs_curr)**2 - safety_dist**2)
    h = (X[0,k]-x_obs_curr)**2 + 25*(X[1,k]-y_obs_curr)**2 - safety_dist**2
    h_dot = 2*(X[0,k]-x_obs_curr)*(X[3,k]*cos(X[2,k])-vx_obs+2) + 2*25*(X[1,k]-y_obs_curr)*(X[3,k]*sin(X[2,k])-vy_obs+2)
    # h_dot = 2*(X[0,k]-x_obs_curr)*(X[3,k]*cos(X[2,k])) + 2*9*(X[1,k]-y_obs_curr)*(X[3,k]*sin(X[2,k]))
    h_dot_dot = 2*(X[3,k]*cos(X[2,k])-vx_obs)**2 + 2*9*(X[3,k]*sin(X[2,k])-vy_obs)**2 \
                + 2*(X[0,k]-x_obs_curr)*(U[0,k]*cos(X[2,k])-X[3,k]*sin(X[2,k])) + \
                + 2*9*(X[1,k]-y_obs_curr)*(U[0,k]*sin(X[2,k])+X[3,k]*cos(X[2,k]))
    g[k] = h_dot + gamma*h + viol_amt[k]
    # g[k+N] = h_dot_dot + 2*gamma*h_dot + gamma*gamma*h + viol_amt2[k]
    U[2,k] = viol_amt[k] #+ viol_amt2[k]
    # U[3,k] = viol_amt[k] #+ viol_amt2[k]

for k in range(N) :
    obj = obj + Q*(P[k+1]-X[4,k])**2 + R*U[1,k]**2 + R2*(60-X[3,k])**2 + R_viol*viol_amt[k]**2

# for k in range(N-1) :
#     obj = obj + R2*(U[k+1] - U[k])**2


OPT_variables = reshape(U,3*N,1)

nlp_prob_B = {'f': obj, 'x': OPT_variables, 'p': P, 'g' : g}
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
lbx_B = np.zeros(3*N)
ubx_B = np.zeros(3*N)

for k in range (1,3*N,3): 
    lbx_B[k]=-0.15
    ubx_B[k]=0.15

for k in range (2,3*N,3): 
    lbx_B[k]=0
    ubx_B[k]=1000

for k in range (0,3*N,3): 
    lbx_B[k]=-1
    ubx_B[k]=1

x0=np.random.rand(3*N)
x0[1::3] = -0.15
def get_optimal_commands(obstacle_data, comm_seq, curr_command, curr_speed) :
    p = [curr_command]
    for i in range(N) :
        p.append(float(comm_seq[i]))
    for obstacle in obstacle_data :
        for val in obstacle[:4] :
            p.append(val)
        break
    p.append(curr_speed)
    
    sB = solverB(x0=x0,p=p,lbx=lbx_B,ubx=ubx_B,lbg=[0]*N)
    x = sB['x']
    u = reshape(x.T,3,N).T
    print("Violation distances : ", u)
    return u
