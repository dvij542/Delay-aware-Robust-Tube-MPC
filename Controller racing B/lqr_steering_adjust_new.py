from casadi import *

import numpy as np
import math
import util_funcs as utils
import params as pars
T = .01
N = 10
safety_dist = 8
K = (1-math.e**(-30*T))/T
gamma = 2
lane_len = 10

x=SX.sym('x')
y=SX.sym('y')
theta=SX.sym('theta')
pedal=SX.sym('pedal')
delta_ac=SX.sym('delta_ac')
v=SX.sym('v')
vperp=SX.sym('vperp')
w=SX.sym('w')
states=vertcat(x,y,theta,v,delta_ac,vperp,w)
delta=SX.sym('delta')
controls=vertcat(pedal,delta)
EPSILON = 0
Ka = 7.4
Kf = -7.4/(83*83)



def calc_force_fy(w,v,vperp,delta,lr_ratio,diff_f):
    alpha = -atan((w*pars.Lf + vperp)/(v)) + delta/pars.steering_ratio
    fz = pars.fz0*pars.Lr/(pars.Lf+pars.Lr) + pars.lift_coeff_f*v**2
    fz0 = 3114#pars.fz0*pars.Lr/(pars.Lf+pars.Lr)
    fzr = fz*(lr_ratio)/(lr_ratio + 1)
    fzl = fz*(1)/(lr_ratio + 1)
    return utils.calc_force_from_slip_angle(-alpha,fzl,fz0,P[-7]) + utils.calc_force_from_slip_angle(-alpha,fzr,fz0,P[-7]) + diff_f   
    

def calc_force_ry(w,v,vperp,lr_ratio,diff_r,Gl,Gr):
    alpha = atan((w*pars.Lr - vperp)/(v))
    fz = pars.fz0*pars.Lf/(pars.Lf+pars.Lr) + pars.lift_coeff_r*v**2
    fz0 = 3114#pars.fz0*pars.Lf/(pars.Lf+pars.Lr)
    fzr = fz*(lr_ratio)/(lr_ratio + 1)
    fzl = fz*(1)/(lr_ratio + 1)
    return Gl*utils.calc_force_from_slip_angle(-alpha,fzl,fz0,P[-7]) + Gr*utils.calc_force_from_slip_angle(-alpha,fzr,fz0,P[-7]) + diff_r   

###########    Model   #####################
P=SX.sym('P',1 + N + 17)
rhs=[
        v*cos(theta+EPSILON) - vperp*sin(theta),
        v*sin(theta+EPSILON) + vperp*cos(theta),
        w,
        Ka*pedal+Kf*v*v,
        K*(delta-delta_ac),
        (calc_force_ry(w,v,vperp,P[-5],P[-1],P[-4],P[-3]) + calc_force_fy(w,v,vperp,delta_ac,P[-5],P[-2])*cos(delta/pars.steering_ratio) - pars.mass*v*w)/pars.mass,
        (P[-6]+calc_force_fy(w,v,vperp,delta,P[-5],P[-2])*pars.Lf*cos(delta/pars.steering_ratio) - calc_force_ry(w,v,vperp,P[-5],P[-1],P[-4],P[-3])*pars.Lr)/pars.moment_of_inertia
    ]                                            
rhs=vertcat(*rhs)
f=Function('f',[states,controls],[rhs])

n_states=7
n_controls=2
U=SX.sym('U',n_controls+2,N)
g=SX.sym('g',2*N)
viol_amt = SX.sym('V',N)
viol_amt2 = SX.sym('VV',N)
# P=SX.sym('P',1 + 2*N + 3 + 5)
X=SX.sym('X',n_states,(N+1))
# X[:,0]=P[0:1]         
X[:3,0] = 0
X[3,0] = P[-10]
X[4,0] = P[0]
X[5,0] = P[-9]
X[6,0] = P[-8]
# X[1] = P[0] + K*(U[0]-P[0])
for k in range(0,N,1):
        st=X[:,k]
        con=U[:-2,k]
        f_value=f(st,con)
        st_next=st+(T*f_value)
        X[:,k+1]=st_next

obj=0
Q=SX(5)
R=SX(0)
R2=SX(1000)
R_viol = 10
R_viol2 = 30
x_obs = P[-14]
y_obs = P[-13]
vx_obs = P[-12]
vy_obs = P[-11]
lambda_ = 2

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
    x,y,theta,v,vperp = X[0,k], X[1,k], X[2,k], X[3,k], X[5,k]
    dlane = P[-17]*x + P[-16]*y + P[-15]
    g[k+N] = 1/lambda_*(P[-17]*(v*cos(theta) - vperp*sin(theta)) + P[-16]*(v*sin(theta) + vperp*cos(theta)) + lambda_*dlane + lambda_*viol_amt2[k])
    # g[k+N] = h_dot_dot + 2*gamma*h_dot + gamma*gamma*h + viol_amt2[k]
    U[2,k] = viol_amt[k] #+ viol_amt2[k]
    U[3,k] = viol_amt2[k] #+ viol_amt2[k]
    # U[3,k] = viol_amt[k] #+ viol_amt2[k]

for k in range(N) :
    obj = obj + Q*(P[k+1]-X[4,k])**2 + R*U[1,k]**2 + R2*(70-X[3,k])**2 + R_viol2*viol_amt[k]**2 + R_viol2*viol_amt2[k]**2

# for k in range(N-1) :
#     obj = obj + R2*(U[k+1] - U[k])**2


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
    lbx_B[k]=-0.15
    ubx_B[k]=0.15

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
x0[1::4] = -0.15

def get_optimal_commands(obstacle_data, comm_seq, curr_command, curr_speed,curr_speed_perp,curr_omega,centre_line_eq,correction_params) :
    print(centre_line_eq)
    p = [curr_command]
    for i in range(N) :
        p.append(float(comm_seq[i]))
    
    p.append(centre_line_eq[0])
    p.append(centre_line_eq[1])
    p.append(centre_line_eq[2])
    
    for obstacle in obstacle_data :
        for val in obstacle[:4] :
            p.append(val)
        break
    p.append(curr_speed)
    p.append(curr_speed_perp)
    p.append(curr_omega)
    for param in correction_params :
        p.append(param)
    lb = [0]*N + [-lane_len]*N
    ub = [math.inf]*N + [3*lane_len]*N
    sB = solverB(x0=x0,p=p,lbx=lbx_B,ubx=ubx_B,lbg=lb,ubg=ub)
    x = sB['x']
    u = reshape(x.T,4,N).T
    print("Violation distances : ", u)
    print("G : ",sB['g'])
    return u
