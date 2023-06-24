#!/usr/bin/env python
import math
import torch.nn as nn
import torch
from torch.optim import Adam,SGD
from casadi import *
import tqdm

# Model parameters
L = 3 # Length of the vehicle
Ka = 4.25 # Ax = Ka*pedal_amount + Kf
Kf = -0.25
mu = 1 # Coefficient of friction
g_constant = 9.8

lr = 0.01 # Learning rate

MAKE_DATASET = True
TRAIN = True 

control_count=0
has_start=True

T = .1 # Time horizon
N = 5 # Number of control intervals

###########   states    ####################
x=SX.sym('x')
y=SX.sym('y')
theta=SX.sym('theta')
v=SX.sym('v')
states=vertcat(x,y,theta,v)
a = SX.sym('a')
delta=SX.sym('delta')
controls=vertcat(a,delta)
EPSILON = 1e-5

# Bicycle model
rhs=[
        v*cos(theta),
        v*sin(theta),
        v*(tan(delta))/L,
        Ka*a+Kf
    ]                               
                                                   
rhs=vertcat(*rhs)
f=Function('f',[states,controls],[rhs])
#print(f)
n_states=4  
n_controls=2
U=SX.sym('U',n_controls,N)
P=SX.sym('P',n_states*2)
g=SX.sym('g',N)
X=SX.sym('X',n_states,(N+1))
X[:,0]=P[0:n_states]         

for k in range(0,N,1):
    st=X[:,k]
    con=U[:,k]
    f_value=f(st,con)
    st_next=st+(T*f_value)
    X[:,k+1]=st_next

ff=Function('ff',[U,P],[X])
obj=0

Q=SX([[10,0 ,0 ,0 ],
      [0 ,10,0 ,0 ],
      [0 ,0 ,0 ,0 ],
      [0 ,0 ,0 ,5 ]])
R=SX([[1,0],
      [0,1]])
R2=SX([[100,0  ],
       [0  ,0  ]])

# Define bjective function for MPC
for k in range(N-1,N,1):
    st=X[:,k+1]
    con=U[:,k]
    obj=obj+(((st- P[n_states:]).T)@Q)@(st- P[n_states:]) + ((con.T)@R)@con

for k in range(0,N-1,1) :
    con=U[:,k]
    con_next = U[:,k+1]
    obj = obj + con.T@R@con + (((con-con_next).T)@R2)@(con-con_next)

opt_variables=vertcat(U)
OPT_variables = reshape(U,2*N,1)

# Constraint Ay (|Ay_max| = mu*g)
for k in range (1,N+1,1): 
    g[k-1] = X[3,k]**2 * tan(U[1,k-1]) / L
    
nlp_prob = {'f': obj, 'x':OPT_variables, 'p': P,'g':g}
options = {
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

solver=nlpsol("solver","ipopt",nlp_prob,options)


lbx=np.zeros(2*N)
ubx=np.zeros(2*N)

for k in range (0,2*N,2): 
    lbx[k]=-1
    ubx[k]=1

for k in range (1,(2*N)-1,2): 
    lbx[k]=-math.pi/4.5
    ubx[k]=math.pi/4.5

u0=np.random.rand(N,n_controls)
x0=reshape(u0,n_controls*N,1)


# Modify this to generate input samples for training according to the requirement
def get_input_set(batch_size) :
    X = torch.rand(batch_size,5)
    vels =  0 + 25*torch.rand(batch_size)
    d = vels*(0.9+0.2*torch.rand(batch_size))
    th = -1 + 2*torch.rand(batch_size)
    dth = th + (-1 + 2*torch.rand(batch_size))/2
    targ_vels =  10 + 15*torch.rand(batch_size)

    X[:,0] = d*np.cos(th)
    X[:,1] = d*np.sin(th)
    X[:,2] = dth
    X[:,3] = vels
    X[:,4] = targ_vels
    return X

def get_gt(x) :
    sols = []
    for i in range(x.shape[0]) :
        dX = float(x[i,0])
        dY = float(x[i,1])
        dTheta = float(x[i,2])
        v_init = float(x[i,3])
        v_target = float(x[i,4])
        x_init = [0,0,0,v_init]
        p = x_init + [dX,dY,dTheta,v_target]
        # print(x0,p,lbx,mu*g_constant)
        s0 = solver(x0=x0,p=p,lbx=lbx,ubx=ubx,ubg=mu*g_constant,lbg=-mu*g_constant)
        out = reshape(s0['x'].T,2,N).T
        # print(out)
        if dY>0 :
            steering = min(min(2*L*dY/(dX**2 + dY**2),2*atan(mu*g_constant*L/(v_init**2))),math.pi/4.5)
        else :
            steering = max(max(2*L*dY/(dX**2 + dY**2),-2*atan(mu*g_constant*L/(v_init**2))),-math.pi/4.5)

        sols.append([float(out[0,0]),float(out[0,1])])
    return torch.tensor(np.array(sols))
    
class model_waypoint(nn.Module):
    def __init__(self,layer_sizes):
        super().__init__()
        self.input_dim = 5
        self.output_dim = 2
        self.layers = []
        self.input = nn.modules.Sequential(nn.Linear(self.input_dim,layer_sizes[0]),nn.ReLU())
        for i in range(len(layer_sizes)-1) :
            self.layers.append(nn.modules.Sequential(nn.Linear(layer_sizes[i],layer_sizes[i+1]),nn.ReLU()))
        self.layers = nn.ModuleList(self.layers)
        self.output = nn.Linear(layer_sizes[-1],self.output_dim)

    def forward(self, x) :
        h = self.input(x)
        for i in range(len(self.layers)) :
            h = self.layers[i](h)
        out = self.output(h)
        return out

def loss_function(y,y_hat) :
    return torch.mean((y-y_hat)**2)

def make_dataset(no_batches,batch_size) :
    X = torch.zeros(no_batches,batch_size,5)
    Y = torch.zeros(no_batches,batch_size,2)
    for i in tqdm.tqdm(range(no_batches)) :
        x = get_input_set(batch_size)
        y = get_gt(x)
        X[i,:,:] = x
        Y[i,:,:] = y
    torch.save(X,'data_X.pkl')
    torch.save(Y,'data_Y.pkl')
    
def main() :
    no_epochs = 50
    batch_size = 256
    no_batches = 200
    print("Making dataset !!!")
    if MAKE_DATASET :
        make_dataset(no_batches,batch_size)
    X = torch.load('data_X.pkl')
    Y = torch.load('data_Y.pkl')
    model = model_waypoint(layer_sizes=[8,16,8,4])
    if TRAIN : 
        model.train()
        optimizer = Adam(model.parameters(), lr=lr)
        # overall_loss = 0
        for n in range(no_epochs):
            for i in range(no_batches) :
                x = X[i,:,:]
                y_pred = model(x)
                y = Y[i,:,:]
                optimizer.zero_grad()
                loss = loss_function(y,y_pred)
                print("Epoch " + str(n) + ", Batch " + str(i) + " loss : ", str(loss.item()))
                loss.backward()
                optimizer.step()
                # print(model(torch.tensor([[5,3],[15,4],[15,-6],[10,10],[10,-10]]).float()))
                # overall_loss += loss.item()
        print("Finished training!!")
        torch.save(model.state_dict(),'trained_model.pt')
    model.eval()
    model.load_state_dict(torch.load('trained_model.pt'))
    print(get_gt(torch.tensor([[10,0,0,10,15]\
                            ,[12,6,0,15,10]\
                            ,[12,-6,0,15,10]\
                            ,[21,6,0.1,25,15]
                            ,[ 0.8145e+00,  1.5676e-02, -2.1214e-03, -1.0081e-02,  1.7714e+01]])))
    print(model(torch.tensor([[10,0,0,10,15]\
                            ,[12,6,0,15,10]\
                            ,[12,-6,0,15,10]\
                            ,[21,6,0.1,25,15]
                            ,[ 0.8145e+00,  1.5676e-02, -2.1214e-03, -1.0081e-02,  1.7714e+01]]).float()))
main()
