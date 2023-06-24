from casadi import *

import numpy as np
import math

T = .1
N = 8
safety_dist = 8
gamma = 2
lane_len = 6

DONT_CONSIDER_STEERING_DYNAMICS = True
if DONT_CONSIDER_STEERING_DYNAMICS :
    K = 1/T
else :
    K = (1-math.e**(-10*T))/T

e1=SX.sym('e1')
e2=SX.sym('e2')
e1_=SX.sym('e1\'')
e2_=SX.sym('e2\'')
delta_ac=SX.sym('delta_ac')
s=SX.sym('s')
vx=SX.sym('vx')
states=vertcat(e1,e1_,e2,e2_,delta_ac,vx,s)

pedal=SX.sym('pedal')
delta=SX.sym('delta')
controls=vertcat(pedal,delta)

EPSILON = 0
Ka = 5 # Pedal constant (F_x = Ka*mass*pedal_value)
Kf = -0.134 # Friction resistance (F_{friction,x} = Kf*mass)
    

###########    Model   #####################
P=SX.sym('P',N + 6 + 1 + 4 + 1)

radius = P[-6]

Cf = 2000
Cr = 2000
lf = 1.5
lr = 1.5
mass = 1600
moi = 1600
steer_to_wheel_ratio = 1#(70*math.pi/180)

if DONT_CONSIDER_STEERING_DYNAMICS : 
    rhs=[
            e1_,
            -e1_*(2*Cf+2*Cr)/(mass*vx) + e2*(2*Cf+2*Cr)/mass + e2_*(-2*Cf*lf+2*Cr*lr)/(mass*vx) + 2*delta*Cf*steer_to_wheel_ratio/mass -(vx*radius)*((2*Cf*lf-2*Cr*lr)/(mass*vx)+vx),
            e2_,
            -e1_*(2*Cf*lf-2*Cr*lr)/(moi*vx) + e2*(2*Cf*lf-2*Cr*lr)/moi -e2_*(2*Cf*lf**2+2*Cr*lr**2)/(moi*vx) + 2*Cf*lf*delta*steer_to_wheel_ratio/moi - (vx*radius)*(2*Cf*lf**2+2*Cr*lr**2)/(moi*vx),
            K*(delta-delta_ac),
            Ka*pedal+Kf*vx,
            vx*cos(e2)-(e1_-vx*e2)*sin(e2),
        ]
else :
    rhs=[
            e1_,
            -e1_*(2*Cf+2*Cr)/(mass*vx) + e2*(2*Cf+2*Cr)/mass + e2_*(-2*Cf*lf+2*Cr*lr)/(mass*vx) + 2*delta_ac*Cf*steer_to_wheel_ratio/mass -(vx*radius)*((2*Cf*lf-2*Cr*lr)/(mass*vx)+vx),
            e2_,
            -e1_*(2*Cf*lf-2*Cr*lr)/(moi*vx) + e2*(2*Cf*lf-2*Cr*lr)/moi -e2_*(2*Cf*lf**2+2*Cr*lr**2)/(moi*vx) + 2*Cf*lf*delta_ac*steer_to_wheel_ratio/moi - (vx*radius)*(2*Cf*lf**2+2*Cr*lr**2)/(moi*vx),
            K*(delta-delta_ac),
            Ka*pedal+Kf*vx,
            vx*cos(e2)-(e1_-vx*e2)*sin(e2),
        ]
    
rhs=vertcat(*rhs)
f=Function('f',[states,controls],[rhs])

n_states=7
n_controls=2
U=SX.sym('U',n_controls+2,N)
g=SX.sym('g',2*N)
viol_amt = SX.sym('V',N)
viol_amt2 = SX.sym('VV',N)
X=SX.sym('X',n_states,(N+1))
X[0,0] = P[N]
X[1,0] = P[N+1]
X[2,0] = P[N+2]
X[3,0] = P[N+3]
X[4,0] = P[N+4]
X[5,0] = P[N+5]
X[6,0] = 0

for k in range(0,N,1):
    st=X[:,k]
    con=U[:-2,k]
    f_value=f(st,con)
    st_next=st+(T*f_value)
    X[:,k+1]=st_next

obj=0
Q=SX(5)
R=SX(0)
# R2=SX(10)
R_viol = 10
R_viol2 = 10
s_obs = P[-5]
e_obs = P[-4]
vs_obs = P[-3]
ve_obs = P[-2]
lambda_ = 100

for k in range(N) :
    h = (X[6,k]-s_obs)**2 + 16*(X[0,k]-e_obs)**2 - safety_dist**2
    h_dot = 2*(X[6,k]-s_obs)*(X[5,k]*cos(X[2,k])-vs_obs) + 2*16*(X[0,k]-e_obs)*(X[1,k]-ve_obs)
    # h_dot_dot = 2*(X[3,k]*cos(X[2,k])-vx_obs)**2 + 2*9*(X[3,k]*sin(X[2,k])-vy_obs)**2 \
    #             + 2*(X[0,k]-x_obs_curr)*(U[0,k]*cos(X[2,k])-X[3,k]*sin(X[2,k])) + \
    #             + 2*9*(X[1,k]-y_obs_curr)*(U[0,k]*sin(X[2,k])+X[3,k]*cos(X[2,k]))
    g[k] = viol_amt[k] #+ h_dot + gamma*h
    # g[k+N] = 1/lambda_*(P[-17]*(v*cos(theta) - vperp*sin(theta)) + P[-16]*(v*sin(theta) + vperp*cos(theta)) + lambda_*dlane + lambda_*viol_amt2[k])
    g[k+N] = 1/lambda_*(lambda_*X[0,k] + X[1,k])+ viol_amt2[k]
    # g[k+N] = h_dot_dot + 2*gamma*h_dot + gamma*gamma*h + viol_amt2[k]
    U[2,k] = viol_amt[k]
    U[3,k] = viol_amt2[k]
    
for k in range(N) :
    obj = obj + Q*(P[k]-X[4,k+1])**2 + R*U[1,k]**2 + Q*(P[-1]-U[0,k])**2 + R_viol2*viol_amt[k]**2 + R_viol2*viol_amt2[k]**2


OPT_variables = reshape(U,4*N,1)

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
lbx_B = np.zeros(4*N)
ubx_B = np.zeros(4*N)

for k in range (1,4*N,4): 
    lbx_B[k]=-0.25
    ubx_B[k]=0.25

for k in range (2,4*N,4): 
    lbx_B[k]=0
    ubx_B[k]=1000

for k in range (3,4*N,4): 
    lbx_B[k]=-1000
    ubx_B[k]=1000

for k in range (0,4*N,4): 
    lbx_B[k]=-1
    ubx_B[k]=1

x0=np.random.rand(4*N)
x0[1::4] = 0.15

def get_optimal_commands(obstacle_data, comm_seq, pedal_seq, curr_command, e1,e1_,e2,e2_,curr_speed,radius,ll_list,lr_list) :
    # print("Input : ",curr_command,comm_seq,e1,e1_)
    p = []
    for i in range(N) :
        p.append(float(comm_seq[i]))
    print("Input : ",e1,obstacle_data[0][1],e1_)
    p.append(e1)
    p.append(e1_)
    p.append(e2)
    p.append(e2_)
    p.append(curr_command)
    p.append(curr_speed)
    p.append(radius)
    
    for obstacle in obstacle_data :
        for val in obstacle[:4] :
            p.append(val)
        break

    p.append(pedal_seq[0])
    lb = [-5]*N + lr_list
    ub = [math.inf]*N + ll_list
    sB = solverB(x0=x0,p=p,lbx=lbx_B,ubx=ubx_B,lbg=lb,ubg=ub)
    x = sB['x']
    u = reshape(x.T,4,N).T
    # print("Violation distances : ", u)
    # print("G : ",sB['g'])
    return u[:,:2]
