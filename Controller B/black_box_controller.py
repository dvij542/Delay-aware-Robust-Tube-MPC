# -*- coding: utf-8 -*-

#!/usr/bin/env python3
from casadi import *

import numpy as np
import math
import rospy
import math
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
import torch.nn as nn
import torch
import lqr_steering_adjust as lqrs

# from ackermann_msgs.msg import AckermannDriveStamped
from prius_msgs.msg import Control

from geometry_msgs.msg import PointStamped
global x_bot
global y_bot
global control_count

################ Hyperparameters ####################
T = .04   # Time horizon
N = 8 # number of control intervals
speed = 10
tolerance = 0.05
save_path_after = 10
noise_to_process_var = 1
wait_time = -1

gt_steering = 0
last_steering_time = 0
control_count=0
has_start=False
curr_time_est = 0.04
curr_time_est_var = 1
var_factor = 0.01
buff_con = [0]*N

scenario = 'dynamic'
time_estimates = []
planned_paths = []
time_to_finish = 0
obstacle_points = np.array([[21,-17],[21,-10],[31,-10],[31,-17]])
inv_set = []
L = 3
Ka = 4.25
Kf = -0.25

DONT_CONSIDER_COMP_DELAY = False
DONT_CONSIDER_STEERING_DYNAMICS = False

###########   states    ####################

x=SX.sym('x')
y=SX.sym('y')
theta=SX.sym('theta')
pedal=SX.sym('pedal')
delta_ac=SX.sym('delta_ac')
v=SX.sym('v')
a = SX.sym('a')
states=vertcat(x,y,theta,v,delta_ac)
delta=SX.sym('delta')
controls=vertcat(a,delta)
ctrl_cmds = np.zeros((N,2))
EPSILON = 0
if DONT_CONSIDER_STEERING_DYNAMICS :
    K = 1/T
else :
    K = (1-math.e**(-30*T))/T

###########    Model   #####################
rhs=[
        v*cos(theta+EPSILON),
        v*sin(theta+EPSILON),
        v*tan(delta_ac+K*(delta-delta_ac)*T/2)/L,
        Ka*a+Kf,
        K*(delta-delta_ac)
    ]                                            
rhs=vertcat(*rhs)
f=Function('f',[states,controls],[rhs])
mu = 1
g_constant = 9.8

n_states=5
n_controls=2
U=SX.sym('U',n_controls,N)
g=SX.sym('g',N+1)
P=SX.sym('P',n_states + 2*N + 3)
X=SX.sym('X',n_states,(N+1))
X[:,0]=P[0:n_states]         


for k in range(0,N,1):
    st=X[:,k]
    con=U[:,k]
    f_value=f(st,con)
    st_next=st+(T*f_value)
    X[:,k+1]=st_next

ff=Function('ff',[U,P],[X])


def dist(a, x, y):
    return (((a.pose.position.x - x)**2) + ((a.pose.position.y - y)**2))**0.5

def path_length_distance(a,b):
    return (((a.pose.position.x - b.pose.position.x)**2) + ((a.pose.position.y - b.pose.position.y)**2))**0.5

def calc_path_length(data):
    # global path_length
    path_length = []
    for i in range(len(data.poses)):
        if i == 0:
            path_length.append(0)
        else:
            path_length.append(path_length[i-1] + path_length_distance(data.poses[i], data.poses[i-1]))
    return path_length

def posCallback(posedata):
    global data,v_bot_init
    data=posedata.pose
    v_bot_init = (posedata.twist.twist.linear.x**2 + posedata.twist.twist.linear.y**2)**0.5

def convert_xyzw_to_rpy(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def pathCallback(pathdata):
    global x_p
    global has_start
    global time_to_finish
    global time_estimates
    global planned_paths
    global N

    if has_start == False :
        ref_path = []
        for pose in pathdata.poses :
            ref_path.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        np.savetxt("ref_path.csv", np.array(ref_path))
        print("Reference path saved")
        time_to_finish = rospy.get_time() + save_path_after
    if rospy.get_time() > time_to_finish :
        np.savetxt('computation_times.csv', time_estimates)
        np.savetxt('paths_planned.csv', planned_paths)
        print("Saved")
    for i in range(N) :
        dist_covered = 0
    has_start=True
    x_p = pathdata
    mpcCallback()

def get_path(x_bot1, y_bot1, x_p, N,speed,T) :  
    out_path = []
    path_lengths = calc_path_length(x_p)
    distances = []    
    for i in range(len(x_p.poses)):
        a = x_p.poses[i]
        distances += [dist(a, x_bot1, y_bot1)]
    # print(distances)
    ep = min(distances)
    total_index=len(x_p.poses)
    cp = distances.index(ep)
    curr_dist = path_lengths[cp]
    i = cp
    
    # In case of very sparsely present points, divide segments into multiple parts to get closest point
    new_dists = []
    if cp > 0 :
        available_length_l = path_lengths[cp] - path_lengths[cp-1]
    else :
        available_length_l = 0
    
    if cp < len(path_lengths) - 1 :
        available_length_r = path_lengths[cp+1] - path_lengths[cp]
    else :
        available_length_r = 0
    
    no_of_segs_l = int(available_length_l/(speed*T)) 
    no_of_segs_r = int(available_length_r/(speed*T)) 
    seg_len_l = available_length_l/max(no_of_segs_l,1)
    seg_len_r = available_length_r/max(no_of_segs_r,1)
    for s in range(no_of_segs_l) :
        x1,y1 = x_p.poses[cp-1].pose.position.x, x_p.poses[cp-1].pose.position.y
        x2,y2 = x_p.poses[cp].pose.position.x, x_p.poses[cp].pose.position.y
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        new_dists += [((xs-x_bot1)**2 + (ys-y_bot1)**2)**0.5]
    new_dists.append(ep)
    for s in range(no_of_segs_r) :
        x1,y1 = x_p.poses[cp].pose.position.x, x_p.poses[cp].pose.position.y
        x2,y2 = x_p.poses[cp+1].pose.position.x, x_p.poses[cp+1].pose.position.y
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        new_dists += [((xs-x_bot1)**2 + (ys-y_bot1)**2)**0.5]
    min_ni = new_dists.index(min(new_dists))
    if min_ni < no_of_segs_l :
        s = min_ni
        x1,y1 = x_p.poses[cp-1].pose.position.x, x_p.poses[cp-1].pose.position.y
        x2,y2 = x_p.poses[cp].pose.position.x, x_p.poses[cp].pose.position.y
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        v1 = x_p.poses[cp-1].pose.position.z
        v2 = x_p.poses[cp].pose.position.z
        vs = v1 + (v2-v1)*(seg_len_l/available_length_l)*(s+1)
        pose_temp=PoseStamped()
        pose_temp.pose.position.x = xs
        pose_temp.pose.position.y = ys
        pose_temp.pose.position.z = vs
        x_p.poses.insert(cp,pose_temp)
        path_lengths.insert(cp,path_lengths[cp-1] + seg_len_l*(s+1))
    if min_ni > no_of_segs_l :
        s = min_ni - no_of_segs_l - 1
        x1,y1 = x_p.poses[cp].pose.position.x, x_p.poses[cp].pose.position.y
        x2,y2 = x_p.poses[cp+1].pose.position.x, x_p.poses[cp+1].pose.position.y
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        v1 = x_p.poses[cp].pose.position.z
        v2 = x_p.poses[cp+1].pose.position.z
        vs = v1 + (v2-v1)*(seg_len_r/available_length_r)*(s+1)
        pose_temp=PoseStamped()
        pose_temp.pose.position.x = xs
        pose_temp.pose.position.y = ys
        pose_temp.pose.position.z = vs
        x_p.poses.insert(cp+1,pose_temp)
        path_lengths.insert(cp+1,path_lengths[cp] + seg_len_r*(s+1))
        cp = cp + 1
    i = cp
    
    # Building the path
    for j in range(N) :
        req_dist = (j+1)*speed*T
        k = i
        while(k<len(path_lengths) and path_lengths[k]-path_lengths[cp]<req_dist ) :
            k += 1
        if k>=len(path_lengths) :
            k = len(path_lengths) - 1
            out_path.append(np.array([x_p.poses[k].pose.position.x,x_p.poses[k].pose.position.y]))
            continue
        a = req_dist + path_lengths[cp] - path_lengths[k-1]
        b = path_lengths[k] - req_dist - path_lengths[cp]
        X1 = np.array([x_p.poses[k-1].pose.position.x,x_p.poses[k-1].pose.position.y,x_p.poses[k-1].pose.position.z])
        X2 = np.array([x_p.poses[k].pose.position.x,x_p.poses[k].pose.position.y,x_p.poses[k].pose.position.z])
        X = X1*b/(a+b) + X2*a/(a+b)
        out_path.append(X)
        i = k-1
    return np.array(out_path)

def get_future_state_est(current_pose,buff_con,curr_time_est) :
    print("Pose before : ", np.array(current_pose))
    x_before, y_before = current_pose[0], current_pose[1]
    print(curr_time_est)
    if DONT_CONSIDER_COMP_DELAY :
        return np.array(current_pose)
    itr = 0
    # print(buff_con,curr_time_est)
    while (curr_time_est > T) :
        # print("a", current_pose,buff_con[itr])
        f_value=f(current_pose,list(buff_con[itr]))
        current_pose = list(np.array(current_pose)+np.array(T*f_value)[:,0])
        itr = itr + 1
        curr_time_est = curr_time_est - T
    # print("b",current_pose,buff_con[itr])
    f_value=f(current_pose,list(buff_con[itr]))
    # print("f_value :", f_value)
    # print(curr_time_est, rospy.get_time())
    print("Pose after :", np.array(current_pose)+np.array(curr_time_est*f_value)[:,0])
    pose_after = np.array(current_pose)+np.array(curr_time_est*f_value)[:,0]
    print("Distance :", ((x_before-pose_after[0])**2+(y_before-pose_after[1])**2)**0.5)
    
    return np.array(current_pose)+np.array(curr_time_est*f_value)[:,0]

def get_next_pose(current_pose,con) :
    # print("Pose before : ", np.array(current_pose))
    # print("c",current_pose,con)
    f_value=f(current_pose,list(con))
    # print(np.array(curr_time_est*f_value)[:,0])
    return np.array(current_pose)+T*np.array(f_value)[:,0]

def update_time_estimates(curr_time_est,curr_time_est_var,meas_time_est):
    curr_time_est_var = curr_time_est_var + 1
    curr_time_est = curr_time_est + (meas_time_est-curr_time_est)*noise_to_process_var/(noise_to_process_var+curr_time_est_var)
    curr_time_est_var = curr_time_est_var*noise_to_process_var/(noise_to_process_var+curr_time_est_var)
    return curr_time_est, curr_time_est_var

def callback_tf(data):
    global gt_steering, last_steering_time
    for tf_data in data.transforms :
        if tf_data.header.frame_id == "chassis" and tf_data.child_frame_id == "fr_axle" :
            # print(tf_data.transform.translation)
            x = tf_data.transform.rotation.x #- init_steering_x
            y = tf_data.transform.rotation.y #- init_steering_y
            z = tf_data.transform.rotation.z #- init_steering_z
            w = tf_data.transform.rotation.w #- init_steering_w
            _,_,gt_steering = convert_xyzw_to_rpy(x,y,z,w)
            last_steering_time = float(str(tf_data.header.stamp))/1e9
            
def mpcCallback():
    t1 = rospy.get_time()
    global curr_time_est
    global time_estimates
    global vel
    global control_count
    global has_start
    global gt_steering
    global curr_time_est_var
    global var_factor
    global buff_con
    global model 
    global ctrl_cmds

    has_start=True
    control_count=0   
    global goal_point
    x_bot_init = data.pose.position.x
    y_bot_init = data.pose.position.y
    siny = +2.0 * (data.pose.orientation.w *
                       data.pose.orientation.z +
                       data.pose.orientation.x *
                       data.pose.orientation.y)
    cosy = +1.0 - 2.0 * (data.pose.orientation.y *
                             data.pose.orientation.y +
                             data.pose.orientation.z *
                             data.pose.orientation.z)
    yaw_car_init = math.atan2(siny,cosy) # yaw in radians
    # print("v_bot_init :", v_bot_init)
    current_pose=[x_bot_init,y_bot_init,yaw_car_init,v_bot_init,gt_steering]
    # print(ctrl_cmds)
    if wait_time < 0 :
        current_pose = get_future_state_est(current_pose,ctrl_cmds,curr_time_est+curr_time_est_var*var_factor*2)
    else :
        current_pose = get_future_state_est(current_pose,ctrl_cmds,wait_time)

    # print("Pose after : ", np.array(current_pose))
    current_pose = list(current_pose)
    start_steering = current_pose[-1]
    startpoint=PointStamped()
    startpoint.header.stamp=rospy.Time.now()
    startpoint.header.frame_id='/map'
    startpoint.point.x=current_pose[0]
    startpoint.point.y=current_pose[1]
    startpoint.point.z=0
    pub4.publish(startpoint)
    ref_poses = get_path(current_pose[0], current_pose[1], x_p, 20, max(0.1,current_pose[3]), T)    
    ctrl_cmds_unprocesed = []
    xs=[ref_poses[-1,0],ref_poses[-1,1],0]

    for i in range(N) :
        dx_ = ref_poses[-1,0] - current_pose[0]
        dy_ = ref_poses[-1,1] - current_pose[1]
        theta_ = current_pose[2]
        dx = dx_*math.cos(theta_) + dy_*math.sin(theta_)
        dy = dy_*math.cos(theta_) - dx_*math.sin(theta_)
        # print(ref_poses[-1,1],ref_poses[-2,1],ref_poses[-1,0],ref_poses[-2,0])
        # print(theta_)
        dtheta = theta_ - math.atan2(ref_poses[-1,1]-ref_poses[-2,1],ref_poses[-1,0]-ref_poses[-2,0])
        # print(dtheta)
        dtheta = float(dtheta)
        # print(dtheta)
        if float(dtheta) > math.pi :
            dtheta -= 2*math.pi
        if float(dtheta) < -math.pi :
            dtheta += 2*math.pi
        v_init = current_pose[3]
        v_target = ref_poses[-1,2]*100
        model.eval()
        # [[ 7.8145e+00,  1.5676e-02, -2.1214e-03, -1.0081e-02,  1.7714e+01]]
        # print(torch.tensor([[dx,dy,dtheta,v_init,v_target]]))
        cmd = model(torch.tensor([[dx,dy,dtheta,v_init,v_target]]).float()).float().detach().numpy()
        print(cmd)
        if dy>0 :
            steering = min(min(2*L*dy/(dx**2 + dy**2),2*atan(mu*g_constant*L/(v_init**2))),math.pi/4.5)
        else :
            steering = max(max(2*L*dy/(dx**2 + dy**2),-2*atan(mu*g_constant*L/(v_init**2))),-math.pi/4.5)

        # cmd[0,1] = steering
        # print("Vals : ",dx,dy,cmd)
        ctrl_cmds_unprocesed.append(cmd[0])
        current_pose = get_next_pose(current_pose,cmd[0])
    if DONT_CONSIDER_STEERING_DYNAMICS :
        ctrl_cmds = np.array(ctrl_cmds_unprocesed)
    else :
        print("Current time, steering time :", rospy.get_time(), last_steering_time)
        # ctrl_cmds_unprocesed = [ctrl_cmds_unprocesed[0]]*4
        print("Commands before :", gt_steering, start_steering, ctrl_cmds_unprocesed)
        ctrl_cmds_unprocesed = np.array(ctrl_cmds_unprocesed)
        # ctrl_cmds_unprocesed[1:,0] = ctrl_cmds_unprocesed[0,0]
        ctrl_cmds_unprocesed[1:,1] = ctrl_cmds_unprocesed[0,1]

        ctrl_cmds = np.array(ctrl_cmds_unprocesed)
        ctrl_cmds[:4,1:2] = np.array(lqrs.get_optimal_commands(ctrl_cmds_unprocesed[:,1],start_steering))
        print("Commands after :", ctrl_cmds)
    ctrlmsg = Path()
    ctrlmsg.header.frame_id='map'
    for i in range(0,4,1):
        pose3=PoseStamped()
        pose3.pose.position.x = min(40, max(-40, float(ctrl_cmds[i,1]) * 180 / math.pi))
        pose3.pose.position.y = t1 + curr_time_est+curr_time_est_var*var_factor*2 + T*i
        pose3.pose.position.z = min(1, max(-1, float(ctrl_cmds[i,0])))
        ctrlmsg.poses.append(pose3)
        
    t2 = rospy.get_time()
    # if (t2-t1)<0.1 :
    if wait_time < 0 :
        rospy.sleep(0.03 + 0.0*math.sin(t2/3.5))#1-(t2-t1))
    else :
        if (t2-t1)<wait_time :
            rospy.sleep(wait_time-(t2-t1))#1-(t2-t1))

    t3 = rospy.get_time()
    new_time_est = t3-t1
    pub1.publish(ctrlmsg)
    if wait_time < 0 :
        if curr_time_est+2*curr_time_est_var*var_factor-new_time_est > 0 :
            rospy.sleep(curr_time_est+2*curr_time_est_var*var_factor-new_time_est)
    print("Time taken : ", new_time_est)
    time_estimates.append([curr_time_est,new_time_est,curr_time_est+2*curr_time_est_var*var_factor,t2])
    
    curr_time_est, curr_time_est_var = update_time_estimates(curr_time_est,curr_time_est_var,new_time_est)
    
    a=PointStamped()
    a.header.stamp=rospy.Time.now()
    a.header.frame_id='/map'
    a.point.x=xs[0]
    a.point.y=xs[1]
    a.point.z=0
    pub2.publish(a)
    # print('lbg',lbg)
    print('Published')

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

def start():
    global pub1
    global pub2
    global pub4
    global model
    model = model_waypoint(layer_sizes=[8,16,4])
    model.load_state_dict(torch.load('MPC/trained_model.pt'))
    model.eval()
    rospy.init_node('path_tracking', anonymous=True)
    pub1 = rospy.Publisher('cmd_delta', Path, queue_size=1)
    pub2 = rospy.Publisher('goal_point', PointStamped, queue_size=1)
    pub4 = rospy.Publisher('start_point', PointStamped, queue_size=1)
    
    rospy.Subscriber("base_pose_ground_truth", Odometry, posCallback,queue_size=1)
    rospy.Subscriber("astroid_path", Path, pathCallback,queue_size=1)
    rospy.Subscriber("/tf", TFMessage, callback_tf,queue_size=1)
    rospy.spin()
    r=rospy.Rate(1000)
    r.sleep()

if __name__ == '__main__':    
    start()
