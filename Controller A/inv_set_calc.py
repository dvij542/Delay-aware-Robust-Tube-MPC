import numpy as np
from pytope import Polytope
import math
import scipy.linalg as la
import matplotlib.pyplot as plt
import tqdm

NX = 5
NU = 2
WB = 4
WK = 3

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

def get_linear_model_matrix(v, phi, delta, DT, without_steering=False):
    
    A = np.matrix(np.zeros((NX, NX)))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB
    A[4, 4] = 1 - WK*DT
    A[3, 4] = DT * v / (WB * math.cos(delta) ** 2)
    
    B = np.matrix(np.zeros((NX, NU)))
    B[2, 0] = DT
    B[4, 1] = WK*DT

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)
    if without_steering :
        
        A = np.matrix(np.zeros((4, 4)))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = DT * math.cos(phi)
        A[0, 3] = - DT * v * math.sin(phi)
        A[1, 2] = DT * math.sin(phi)
        A[1, 3] = DT * v * math.cos(phi)
        A[3, 2] = DT * math.tan(delta) / WB
        
        
        B = np.matrix(np.zeros((4, NU)))
        B[2, 0] = DT
        B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)
        
        C = np.zeros(4)
        C[0] = DT * v * math.sin(phi) * phi
        C[1] = - DT * v * math.cos(phi) * phi
        C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)
        
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
    A,B,C = get_linear_model_matrix(v, theta, 0, DT, without_steering)
    w = Polytope(lb=(-0.01,-0.01,-0.01,-0.01,-0.01), ub=(0.01,0.01,0.01,0.01,0.01))
    if without_steering :
        w = Polytope(lb=(-0.01,-0.01,-0.01,-0.01), ub=(0.01,0.01,0.01,0.01))

    K, _, _ = dlqr(A,B,Q,R)
    Ak = A - B * K
    Z = w
    Mul = np.matrix(np.diag(np.array([1,1,1,1,1])))
    if without_steering :
        Mul = np.matrix(np.diag(np.array([1,1,1,1])))

    for i in range(N) :
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

def get_K(DT,theta,v,Q,R,N,without_steering = False) :
    A,B,C = get_linear_model_matrix(v, theta, 0, DT,without_steering)
    w = Polytope(lb=(-0.01,-0.01,-0.01,-0.01,-0.01), ub=(0.01,0.01,0.01,0.01,0.01))
    K, _, _ = dlqr(A,B,Q,R)
    Ak = A - B * K
    return K

def get_inv_set_test(DT,theta,phi,v,Q,R,N) :
    A,B,C = get_linear_model_matrix(v, theta, phi, DT,without_steering=True)
    w = Polytope(lb=(-0.01,-0.01,-0.01,-0.01), ub=(0.01,0.01,0.01,0.01))
    K, _, _ = dlqr(A,B,Q,R)
    Ak = A - B * K
    Z = w
    Mul = np.matrix(np.diag(np.array([1,1,1,1])))
    for i in range(4) :
        # print(i)
        Mul = Mul*Ak 
        Z = Z + Mul*w
    angles = np.arctan2(Z.V[:,1],Z.V[:,0])
    projected_verts = np.unique(Z.V[np.argsort(angles),:2],axis=0)
    projected_Z = Polytope(projected_verts)
    
    Uc = Polytope(lb=(-4,-0.75),ub=(4,0.75))
    x_min = np.min(projected_Z.V[:,0])
    x_max = np.max(projected_Z.V[:,0])
    y_min = np.min(projected_Z.V[:,1])
    y_max = np.max(projected_Z.V[:,1])
    r = (x_max**2 + y_max**2)**0.5
    Z = np.matrix(np.array(np.diag([r/x_max,r/y_max,1,1]))) * Z
    Uc_robust = Uc - K*Z
    # plt.scatter(projected_verts[:,0],projected_verts[:,1])
    # print(y_min,y_max)
    # plt.show()

    # print(Uc_robust.V)
    return x_max,y_max,abs(Uc_robust.V[0,0]),abs(Uc_robust.V[0,1])

# x_maxess = []
# y_maxess = []
# a_maxess = []
# steering_maxess = []
# fig, ax = plt.subplots()
# for phi in np.arange(0,0.5,0.1) :
#     x_maxes = []
#     y_maxes = []
#     a_maxes = []
#     steering_maxes = []
#     for vel in tqdm.tqdm(np.arange(1,10,0.5)) :
#         x_max,y_max,a_max,steering_max = get_inv_set_test(0.1,0,phi,vel,np.matrix(np.diag(np.array([1,1,1,1]))),np.matrix(np.diag(np.array([0.1,0.1]))),8)
#         x_maxes.append(x_max)
#         y_maxes.append(y_max)
#         a_maxes.append(a_max)
#         steering_maxes.append(steering_max)
#     x_maxess.append(x_maxes)
#     y_maxess.append(y_maxes)
#     plt.plot(np.arange(1,10,0.5),x_maxes, label='x-lim for phi='+str(phi))
#     plt.plot(np.arange(1,10,0.5),y_maxes, label='y-lim for phi='+str(phi))

# plt.legend()
# plt.xlabel("Speed")
# plt.ylim([0,0.1])
# # print(steering_maxes)
# # print(a_maxes)
# # plt.plot(np.arange(0,0.5,0.1),a_maxes)
# # plt.plot(np.arange(0,0.5,0.1),steering_maxes)

# plt.show()