import numpy as np
from pygame.constants import K_a, K_f
from pytope import Polytope
import math
import scipy.linalg as la
import matplotlib.pyplot as plt
import tqdm

NX = 6
NU = 2
WB = 4
WK = 11

Cf = 4000
Cr = 4000
lf = 1.5
lr = 1.5
mass = 1600
moi = 1600
Ka = 5.3
Kf = -0.174
WITHOUT_STEERING = True

def calc_inv_set(dt,N,v,L,Sl) :
    A = np.array([[1,v*dt],[0,1]])
    B = np.array([[0],[v*dt/L]])
    w = Polytope(lb=(-0.001,-0.01), ub=(0.001,0.01))
    Q = np.array([[1,0],[0,1]])
    R = 0.1
    Ak = np.array([[1,0.5],[-0.2725,0.3391]])
    Mul = np.array([[1,0],[0,1]])
    K = [[-2.1802],[-5.2876]]
    W = w
    for i in range(N) :
        Mul = np.matmul(Mul,Ak) 
        W = W + Mul*w
    y_min = np.min(W.V[:,0])
    y_max = np.max(W.V[:,1])
    inv_set = Polytope(lb=(-0.05,2*y_min), ub=(0.05,2*y_max))
    return inv_set

def get_new_X_constraint(dt,N,v,L,Sl,old_v,theta) :
    inv_set = calc_inv_set(dt,N,v,L,Sl)
    old_set = Polytope(lb=(old_v[0][0],old_v[0][1]), ub=(old_v[-1][0],old_v[-1][1]))
    rot = theta
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
                     [np.sin(rot), np.cos(rot)]])
    inv_set = rot_mat*inv_set
    new_set = inv_set + old_set
    x_min = np.min(new_set.V[:,0])
    x_max = np.max(new_set.V[:,0])
    y_min = np.min(new_set.V[:,1])
    y_max = np.max(new_set.V[:,1])
    return [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]]

def get_new_X_constraint_fast(inv_set,old_v,theta) :
    rot = theta
    old_set = Polytope(lb=(old_v[0][0],old_v[0][1]), ub=(old_v[2][0],old_v[2][1]))
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
                     [np.sin(rot), np.cos(rot)]])
    inv_set = rot_mat*inv_set
    new_set = inv_set + old_set
    x_min = np.min(new_set.V[:,0])
    x_max = np.max(new_set.V[:,0])
    y_min = np.min(new_set.V[:,1])
    y_max = np.max(new_set.V[:,1])
    return [[x_min,y_min],[x_min,y_max],[x_max,y_max],[x_max,y_min]]

def get_linear_model_matrix(v, curvature, vy, phi, DT, without_steering=False):
    A = np.matrix(np.zeros((NX, NX)))
    A[0, 1] = 1.
    A[1, 1] = - 2*(Cf+Cr)/(mass*v)
    A[1, 2] = 2*(Cf+Cr)/mass
    A[1, 3] = (-2*Cf*lf+2*Cr*lr)/(mass*v)
    A[1, 5] = 2*(Cf)/mass
    A[2, 3] = 1.
    A[3, 1] = - (2*Cf*lf-2*Cr*lr) / (moi*v)
    A[3, 2] = 2*(Cf*lf-Cr*lr)/moi
    A[3, 3] = - (2*Cf*lf**2 + 2*Cr*lr**2)/(moi*v)
    A[3, 5] = 2*Cf*lf/moi
    A[1 ,4] = curvature*(-(2*Cf*lf-2*Cr*lr)/(mass*v)-v)
    A[3, 4] = -curvature*(2*Cf*lf**2+2*Cr*lr**2)/(moi*v)
    A[4, 4] = Kf
    A[5, 5] = -WK
    A = np.matrix(np.identity(NX)) + A*DT

    B = np.matrix(np.zeros((NX, NU)))
    B[4, 0] = Ka
    B[5, 1] = WK

    B = B*DT

    C = np.zeros(NX)
    # phi = 0
    print(without_steering)
    if without_steering :
        # print("aya")
        A = np.matrix(np.zeros((NX-1, NX-1)))
        fact_1 = 1/(1+((vy+lf*phi)/v)**2)
        fact_2 = 1/(1+((vy-lr*phi)/v)**2)
        A[0, 1] = 1.
        A[1, 1] = - 2*Cf*fact_1/(mass*v)+2*Cr*fact_2/(mass*v)
        A[1, 2] = 2*(Cf*fact_1+Cr*fact_2)/mass
        A[1, 3] = (-2*Cf*fact_1*lf+2*Cr*fact_2*lr)/(mass*v)
        A[2, 3] = 1.
        A[3, 1] = - (2*Cf*fact_1*lf-2*Cr*fact_2*lr) / (moi*v)
        A[3, 2] = 2*(Cf*fact_1*lf-Cr*fact_2*lr)/moi
        A[3, 3] = - (2*Cf*fact_1*lf**2 + 2*Cr*fact_2*lr**2)/(moi*v)
        A[1 ,4] = curvature*(-(2*Cf*fact_1*lf-2*Cr*fact_2*lr)/(mass*v)-v)
        A[3, 4] = -curvature*(2*Cf*fact_1*lf**2+2*Cr*fact_2*lr**2)/(moi*v)
        A[4, 4] = Kf
        A = np.matrix(np.identity(NX-1)) + A*DT

        B = np.matrix(np.zeros((NX-1, NU)))
        B[3, 1] = 2*Cf*lf/moi
        B[4, 0] = Ka
        B[1, 1] = 2*(Cf)/mass
        
        B = B*DT

        C = np.zeros(NX-1)
        # print(A,B,C)

    return A, B, C

def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE) by iterative method
    """
    P = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Pn = A.T * P * A - A.T * P * B * la.inv(R + B.T * P * B) * B.T * P * A + Q
        if (abs(Pn - P)).max() < eps:
            P = Pn
            break
        P = Pn
    return Pn

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)
    K = np.matrix(la.inv(B.T * X * B + R) * (B.T * X * A))  
    eigVals, eigVecs = la.eig(A - B * K)

    return K, X, eigVals

def get_inv_set(DT,theta,v,Q,R,N,without_steering=False) :
    # print(without_steering)
    A,B,C = get_linear_model_matrix(v, theta, 0, 0, DT, without_steering)
    w = Polytope(lb=(-0.01,-0.01,-0.01,-0.01,-0.01, -0.01), ub=(0.01,0.01,0.01,0.01,0.01, 0.01))
    if without_steering :
        w = Polytope(lb=(-0.01,-0.01,-0.01,-0.01,-0.01), ub=(0.01,0.01,0.01,0.01,0.01))

    # print(A,B,Q,R)
    K, _, _ = dlqr(A,B,Q,R)
    Ak = A - B * K
    Z = w
    Mul = np.matrix(np.diag(np.array([1,1,1,1,1,1])))
    if without_steering :
        Mul = np.matrix(np.diag(np.array([1,1,1,1,1])))

    for i in range(3) :
        Mul = Mul*Ak 
        Z = Z + Mul*w
    angles = np.arctan2(Z.V[:,1],Z.V[:,0])
    projected_verts = np.unique(Z.V[np.argsort(angles),:2],axis=0)
    projected_Z = Polytope(projected_verts)
    
    Uc = Polytope(lb=(-4,-0.75),ub=(4,0.75))
    x_max = np.max(projected_Z.V[:,0])
    y_max = np.max(projected_Z.V[:,1])
    r = (x_max**2 + y_max**2)**0.5
    inv_set = Polytope(lb=(-r,-r), ub=(r,r))
    return inv_set

def get_K(DT,curvature,v,Q,R,N,without_steering = True) :
    A,B,_ = get_linear_model_matrix(v, curvature, 0, DT,without_steering)
    K, _, _ = dlqr(A,B,Q,R)
    return K

def get_inv_set_test(DT,curvature,vy,phi,v,Q,R,N) :
    A,B,_ = get_linear_model_matrix(v, curvature, vy, phi, DT, without_steering=True)
    if WITHOUT_STEERING :
        w = Polytope(lb=(-0.01,-0.01,-0.01,-0.01,-0.01), ub=(0.01,0.01,0.01,0.01,0.01))
    else :    
        w = Polytope(lb=(-0.01,-0.01,-0.01,-0.01,-0.01,-0.01), ub=(0.01,0.01,0.01,0.01,0.01,0.01))
    # print("A=",A)
    # print("B=",B)
    # print("Q=",Q)
    # print("R=",R)

    K, _, _ = dlqr(A,B,Q,R)
    Ak = A - B * K
    Z = w
    # print(B,K)
    if WITHOUT_STEERING :
        Mul = np.matrix(np.diag(np.array([1,1,1,1,1])))
    else :
        Mul = np.matrix(np.diag(np.array([1,1,1,1,1,1])))
    
    # print(Ak)
    for i in range(3) :
        # print(i,len(Z.V))
        Mul = Mul*Ak 
        Z = Z + Mul*w
    # angles = np.arctan2(Z.V[:,1],Z.V[:,0])
    # projected_verts = np.unique(Z.V[np.argsort(angles),:2],axis=0)
    # projected_Z = Polytope(projected_verts)
    
    Uc = Polytope(lb=(-1,-0.75),ub=(1,0.75))
    e1_min = np.min(Z.V[:,0])
    e1_max = np.max(Z.V[:,0])
    e2_min = np.min(Z.V[:,2])
    e2_max = np.max(Z.V[:,2])
    v_max = np.max(Z.V[:,4])
    # r = (x_max**2 + y_max**2)**0.5
    # Z = np.matrix(np.array(np.diag([r/x_max,r/y_max,1,1]))) * Z
    Uc_robust = Uc - K*Z
    # plt.scatter(projected_verts[:,0],projected_verts[:,1])
    # print(y_min,y_max)
    # plt.show()
    # print(Z.V[:,0])
    # print(Uc_robust.V)
    return e1_max,e2_max,v_max*3*DT,abs(Uc_robust.V[0,0]),abs(Uc_robust.V[0,1])

if WITHOUT_STEERING : 
    Q_robust = np.matrix(np.diag(np.array([.1,.1,.1,.1,.1])))
    R_robust = np.matrix(np.diag(np.array([1,1])))
else :
    Q_robust = np.matrix(np.diag(np.array([.1,.1,.1,.1,.1,.1])))
    R_robust = np.matrix(np.diag(np.array([1,1])))
N = 8

# print(get_K(0.1,0,20,Q_robust,R_robust,N))
# vy = 5
# phi = 0.5
# e1_maxess = []
# e2_maxess = []
# s_maxess = []
# steering_maxess = []
# a_maxess = []
# fig, ax = plt.subplots()
# for curvature in np.arange(0,0.09,0.03) :
#     e1_maxes = []
#     e2_maxes = []
#     s_maxes = []
#     a_maxes = []
#     steering_maxes = []
#     for vel in tqdm.tqdm(np.arange(10,30,2)) :
#         e1_max,e2_max,s_max,a_max,steering_max = get_inv_set_test(0.05,curvature,vy,phi,vel,np.matrix(np.diag(np.array([1]*5))),np.matrix(np.diag(np.array([0.1,0.1]))),8)
#         e1_maxes.append([vel,e1_max])
#         e2_maxes.append([vel,e2_max])
#         s_maxes.append([vel,s_max])
#         a_maxes.append([vel,a_max])
#         steering_maxes.append([vel,steering_max])
#     e1_maxess.append(e1_maxes)
#     e2_maxess.append(e2_maxes)
#     s_maxess.append(s_maxes)
#     steering_maxess.append(steering_maxes)
#     a_maxess.append(a_maxes)
#     # print(e1_maxes)
#     plt.plot(np.array(e1_maxes)[:,0],np.array(e1_maxes)[:,1], label='e1-lim for curvature='+str(curvature))
#     plt.plot(np.array(e2_maxes)[:,0],np.array(e2_maxes)[:,1], label='e2-lim for curvature='+str(curvature))
#     plt.plot(np.array(s_maxes)[:,0],np.array(s_maxes)[:,1], label='s-lim for curvature='+str(curvature))

#     # plt.plot(np.array(a_maxes)[:,0],np.array(a_maxes)[:,1], label='a-lim for curvature='+str(curvature))
#     # plt.plot(np.array(steering_maxes)[:,0],np.array(steering_maxes)[:,1], label='steering-lim for curvature='+str(curvature))
#     # plt.show()

# plt.legend()
# plt.xlabel("Speed")
# # plt.ylim([0,0.08])
# plt.ylabel("Lim")
# plt.title('vy : ' + str(vy) + " omega : " + str(phi))
# # print(steering_maxes)
# # print(a_maxes)
# # # plt.plot(np.arange(0,0.5,0.1),a_maxes)
# # # plt.plot(np.arange(0,0.5,0.1),steering_maxes)
# e1_maxess = np.array(e1_maxess)
# e2_maxess = np.array(e2_maxess)
# s_maxess = np.array(s_maxess)
# steering_maxess = np.array(steering_maxess)
# a_maxess = np.array(a_maxess)
# plt.savefig('outputs/errors_4.png')
# plt.show()
# np.savetxt('outputs/e1_maxess.csv',e1_maxess.reshape(e1_maxess.shape[0],e1_maxess.shape[1]*e1_maxess.shape[2]))
# np.savetxt('outputs/e2_maxess.csv',e2_maxess.reshape(e1_maxess.shape[0],e1_maxess.shape[1]*e1_maxess.shape[2]))
# np.savetxt('outputs/s_maxess.csv',s_maxess.reshape(e1_maxess.shape[0],e1_maxess.shape[1]*e1_maxess.shape[2]))
# np.savetxt('outputs/a_maxess.csv',a_maxess.reshape(e1_maxess.shape[0],e1_maxess.shape[1]*e1_maxess.shape[2]))
# np.savetxt('outputs/steering_maxess.csv',steering_maxess.reshape(e1_maxess.shape[0],e1_maxess.shape[1]*e1_maxess.shape[2]))
