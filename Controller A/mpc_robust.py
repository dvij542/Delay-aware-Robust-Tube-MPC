# -*- coding: utf-8 -*-

#!/usr/bin/env python3

from adaptive_kalman_filter import update_delay_time_estimate
from casadi import *
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import Point32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
from IRIS import *
from inv_set_calc import *
from prius_msgs.msg import Control
from adaptive_kalman_filter import *
from geometry_msgs.msg import PointStamped

import numpy as np
import math
import rospy
import math

################ Parameters ####################
T = .1     # Time horizon
N = 8 # number of control intervals
speed = 5
save_path_after = 10
wait_time = -1 # If -1, adaptive kalman filter will be used to predict it
Td = 0.0 # Actuator processing delay
gt_steering = 0 # Variable to communicate current value of steering angle

has_start=False
curr_time_est = 0  # Estimated mean value of computation time
curr_time_est_var = 0 # Estimated variance of computation time

buff_con = [0]*N # buffer sequence of commands
g_const = 9.8
mu = 1 # Friction constant (F_{y,max} = mu*mass*g_const)
scenario = 'static'


time_estimates = []
planned_paths = []
time_to_finish = 0
obstacle_points = np.array([[22,-18],[22,-10],[32,-10],[32,-18]]) # Static obstacle
inv_set = [] # Invariant set, Z

L = 3 # Length of the vehicle in m
Ka = 4.25 # Pedal constant (F_x = Ka*mass*pedal_value)
Kf = -0.25 # Friction resistance (F_{friction,x} = Kf*mass)
vehicle_footprint = [[3,1],[-1,1],[-1,-1],[3,-1]] # Vehicle dimensions in m

Q_robust = np.matrix(np.diag(np.array([1,1,1,1]))) 
R_robust = np.matrix(np.diag(np.array([.1,.1])))


############ Calculated offline from inv_set_calc,py (Last commented part) ###############
steering_limits = [0.5511084632063113, 0.5511084632063113, 0.5511084632063113, \
    0.5185237587941808, 0.5511084632063113, 0.5727896385850489, 0.5896766286658156, \
    0.6037252485785009, 0.616120511291855, 0.6266117297066048, 0.6266117297066048]
acc_limits = [3.7014163903050914, 3.700715491966788, 3.7157664426919617, \
    3.7346625840889347, 3.751783194067104, 3.7654178037240746, 3.7756355027001733, \
    3.7829216295990125, 3.7880532616963265, 3.791426044016998, 3.791426044016998]
#################### Function of speeds (step size 1 m/s) ################################


DONT_CONSIDER_COMP_DELAY = False # If True, computation delay compensation will not be considered
DONT_CONSIDER_STEERING_DYNAMICS = False # If True, actuator dynamic delay compensation will not be considered

FIRST_TIME = True

# Definitions of optimization objective 
if True : 
    ###########   States    ####################

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
    if DONT_CONSIDER_STEERING_DYNAMICS :
        K = 1/T
    else :
        K = (1-math.e**(-30*T))/T

    ###########    Model   #####################
    rhs=[
            v*cos(theta+EPSILON),
            v*sin(theta+EPSILON),
            v*tan(delta_ac+K*(delta-delta_ac)*T/2)/L,
            Ka*pedal+Kf,
            K*(delta-delta_ac)
        ]                                            
    rhs=vertcat(*rhs)
    f=Function('f',[states,controls],[rhs])

    n_states=5
    n_controls=2
    U=SX.sym('U',n_controls,N)
    g=SX.sym('g',(1+len(vehicle_footprint))*(N+1))
    P=SX.sym('P',n_states + 2*N + 3)
    X=SX.sym('X',n_states,(N+1))
    X[:,0]=P[0:n_states]         


    for k in range(0,N,1):
        st=X[:,k]
        con=U[:,k]
        f_value=f(st,con)
        st_next=st+(T*f_value)
        X[:,k+1]=st_next

    ############### Objective function ################

    ff=Function('ff',[U,P],[X])

    obj=0
    Q=SX([[5,0],
        [0,5]])
    Q_speed = 10

    R=SX([[1,0],
        [0,1]])

    R2=SX([[10,0],
        [0,10]])

    for k in range(0,N,1):
        st=X[:,k+1]
        con=U[:,k]
        j = n_states + 2*k
        obj=obj+(((st[:2]- P[j:(j+2)]).T)@Q)@(st[:2]- P[j:(j+2)]) + Q_speed*(st[3]-speed)**2 + con.T@R@con

    for k in range(0,N-1,1):
        prev_con=U[:,k]
        next_con=U[:,k+1]
        obj=obj+(prev_con- next_con).T@R2@(prev_con- next_con)

    OPT_variables = reshape(U,2*N,1)

    a_eq = P[-3]
    b_eq = P[-2]
    c_eq = P[-1]

    for v in range(0,len(vehicle_footprint)) :
        for k in range (0,N+1,1): 
            g[k+(N+1)*v] = a_eq*(X[0,k]+vehicle_footprint[v][0]*cos(X[2,k])\
                -vehicle_footprint[v][1]*sin(X[2,k])) + b_eq*(X[1,k]+\
                vehicle_footprint[v][0]*sin(X[2,k]) + \
                vehicle_footprint[v][1]*cos(X[2,k])) + c_eq   

    for k in range (0,N+1,1): 
        g[k+(N+1)*len(vehicle_footprint)] = -X[3,k]**2*tan(X[2,k])/L

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

    u0=np.random.rand(N,2)
    x0=reshape(u0,2*N,1)

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

# Called when position is updated
def posCallback(posedata):
    global data,v_bot_init
    data=posedata.pose
    v_bot_init = (posedata.twist.twist.linear.x**2 + posedata.twist.twist.linear.y**2)**0.5
    if has_start :
        mpcCallback()

# Called when dynamic obstacle position is updated
def obsPosCallback(posedata) :
    global obstacle_points
    x_obst = posedata.pose.pose.position.x
    y_obst = posedata.pose.pose.position.y 
    _,_,yaw = convert_xyzw_to_rpy(posedata.pose.pose.orientation.x,posedata.pose.pose.orientation.y,posedata.pose.pose.orientation.z,posedata.pose.pose.orientation.w)
    rot_mat = np.array([[math.cos(yaw),math.sin(yaw)],[math.sin(-yaw),math.cos(yaw)]])
    c_obst = np.array([[x_obst,y_obst]])
    dims = np.array([[-1,-1],[-1,1],[3,1],[3,-1]])
    obstacle_points = c_obst + np.matmul(dims,rot_mat)
    obs_draw = PolygonStamped()
    obs_draw.header.frame_id='map'
    # obstacle_points1 = np.array([[22,-17],[22,-9],[33,-9],[33,-17]])
    if scenario == 'dynamic' :
        for p in obstacle_points :
            temp_point = Point32()
            temp_point.x = p[0]
            temp_point.y = p[1]
            obs_draw.polygon.points.append(temp_point)
        pub_obs.publish(obs_draw)
            
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

# Called whe transforms are updated, gets the updated value of actual steering
def callback_tf(data):
    global gt_steering
    for tf_data in data.transforms :
        if tf_data.header.frame_id == "chassis" and tf_data.child_frame_id == "fr_axle" :
            # print(tf_data.transform.translation)
            x = tf_data.transform.rotation.x #- init_steering_x
            y = tf_data.transform.rotation.y #- init_steering_y
            z = tf_data.transform.rotation.z #- init_steering_z
            w = tf_data.transform.rotation.w #- init_steering_w
            r,p,yaw = convert_xyzw_to_rpy(x,y,z,w)
            gt_steering = yaw
            print("Updated GT_steering, ", gt_steering)
            obs_draw = PolygonStamped()
            obs_draw.header.frame_id='map'
            # obstacle_points1 = np.array([[22,-17],[22,-9],[33,-9],[33,-17]])
            if scenario == 'static' :
                for p in obstacle_points :
                    temp_point = Point32()
                    temp_point.x = p[0]
                    temp_point.y = p[1]
                    obs_draw.polygon.points.append(temp_point)
                pub_obs.publish(obs_draw)
                # print("Obstacle mark published")

# Called when new path is received from planner
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
            ref_path.append([pose.pose.position.x, pose.pose.position.y])
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
    # mpcCallback()

# Preprocessing to get the trackable path by the vehicle (for MPC) at current speed, for N steps at T step length
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
    
    no_of_segs_l = 3*int(available_length_l/(speed*T)) 
    no_of_segs_r = 3*int(available_length_r/(speed*T)) 
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
    min_ni += 1
    if min_ni < no_of_segs_l :
        s = min_ni
        x1,y1 = x_p.poses[cp-1].pose.position.x, x_p.poses[cp-1].pose.position.y
        x2,y2 = x_p.poses[cp].pose.position.x, x_p.poses[cp].pose.position.y
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        pose_temp=PoseStamped()
        pose_temp.pose.position.x = xs
        pose_temp.pose.position.y = ys
        x_p.poses.insert(cp,pose_temp)
        path_lengths.insert(cp,path_lengths[cp-1] + seg_len_l*(s+1))
    if min_ni > no_of_segs_l :
        s = min_ni - no_of_segs_l - 1
        x1,y1 = x_p.poses[cp].pose.position.x, x_p.poses[cp].pose.position.y
        x2,y2 = x_p.poses[cp+1].pose.position.x, x_p.poses[cp+1].pose.position.y
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        pose_temp=PoseStamped()
        pose_temp.pose.position.x = xs
        pose_temp.pose.position.y = ys
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
        X1 = np.array([x_p.poses[k-1].pose.position.x,x_p.poses[k-1].pose.position.y])
        X2 = np.array([x_p.poses[k].pose.position.x,x_p.poses[k].pose.position.y])
        X = X1*b/(a+b) + X2*a/(a+b)
        out_path.append(X)
        i = k-1
    return np.array(out_path)

# Get shifted initial pose of the ego vehicle
def get_future_state_est(current_pose,buff_con,curr_time_est) :
    # print("Pose before : ", np.array(current_pose))
    if DONT_CONSIDER_COMP_DELAY :
        return np.array(current_pose)
    itr = 0
    while (curr_time_est > T) :
        # print("Start ", buff_con)
        if buff_con[0] !=0 :
            # print("Was ", buff_con[itr,:])
            f_value=f(current_pose,buff_con[itr,:])
        else :
            f_value=f(current_pose,float(buff_con[itr]))
        current_pose = list(np.array(current_pose)+np.array(T*f_value)[:,0])
        itr = itr + 1
        curr_time_est = curr_time_est - T
    # print("Was ", buff_con[itr])
    # print("Used ", float(buff_con[itr]))
    if buff_con[0] !=0 :
        # print("Was ", buff_con[itr,:])
        f_value=f(current_pose,buff_con[itr,:])
    else :
        f_value=f(current_pose,float(buff_con[itr]))# print("f_value :", f_value)
    # print(curr_time_est, rospy.get_time())
    # print(np.array(curr_time_est*f_value)[:,0])
    return np.array(current_pose)+np.array(curr_time_est*f_value)[:,0]

# IMPORTANT : MPC for optimization
def mpcCallback():
    t1 = rospy.get_time()
    global curr_time_est, curr_time_est_var
    global time_estimates
    global gt_steering
    global has_start
    global buff_con
    
    has_start=True
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
    print("v_bot_init :", v_bot_init)
    current_pose=[x_bot_init,y_bot_init,yaw_car_init,v_bot_init,gt_steering]
    
    # Shift initial state according to the current upper bound on time estimate
    if wait_time < 0 :
        current_pose = get_future_state_est(current_pose,buff_con,curr_time_est+curr_time_est_var*beta+Td)
    else :
        current_pose = get_future_state_est(current_pose,buff_con,wait_time)
    # print("Pose after : ", np.array(current_pose))

    # Inflate the obstacles according to invariant set, Z
    obstacle_points_updated = get_new_X_constraint_fast(inv_set,obstacle_points,current_pose[2])
    # print(obstacle_points, obstacle_points_updated)
    
    # IRIS constraint line for each inflated obstacle
    a_eqn, b_eqn, c_eqn, i_eqn = get_safety_line_eqn(obstacle_points_updated,current_pose[0],current_pose[1])
    
    # Plot shifted start state
    current_pose = list(current_pose)
    startpoint=PointStamped()
    startpoint.header.stamp=rospy.Time.now()
    startpoint.header.frame_id='/map'
    startpoint.point.x=current_pose[0]
    startpoint.point.y=current_pose[1]
    startpoint.point.z=0
    pub4.publish(startpoint)
    # print(current_pose)
    
    # Get path 
    ref_poses = get_path(current_pose[0], current_pose[1], x_p, N, min(current_pose[3]+1,speed), T)    
    
    # End point of path for plotting
    xs=[ref_poses[-1,0],ref_poses[-1,1],0]
    
    p=current_pose+list(ref_poses.reshape(2*N))+[a_eqn,b_eqn,c_eqn]
    
    # Steering and acctuator limits
    lbx=np.zeros(2*N)
    ubx=np.zeros(2*N)
    for k in range (1,2*N,2): 
        # print("Steering limits :", steering_limits[int(v_bot_init)])
        lbx[k] = -steering_limits[int(v_bot_init)]
        ubx[k] = steering_limits[int(v_bot_init)]

    for k in range (0,2*N,2): 
        # print("Acc limits :", acc_limits[int(v_bot_init)])
        lbx[k] = -acc_limits[int(v_bot_init)]/4
        ubx[k] = acc_limits[int(v_bot_init)]/4

    # Solve optimization problem
    so=solver(x0=x0,p=p,lbx=lbx,ubx=ubx,lbg=([0]*(len(vehicle_footprint)*(N+1))+[-mu*g_const]*(N+1))) 
    x=so['x']
    u = reshape(x.T,2,N).T        
    buff_con = u[:,:]
    ff_value=ff(u.T,p).T
    
    ctrlmsg = Path()
    ctrlmsg.header.frame_id='map'
    # print('current_pose',current_pose)
    
    reference_trajectory=Path()
    reference_trajectory.header.frame_id='map'
    predicted_trajectory=Path()
    predicted_trajectory.header.frame_id='map'
    boundary_line=Path()
    boundary_line.header.frame_id='map'
    for i in range(0,N,1):
        # For plotting
        pose=PoseStamped()
        pose.pose.position.x=ff_value[i,0]
        a=pose.pose.position.x=ff_value[i,0]
        pose.pose.position.y=ff_value[i,1]
        pose.pose.orientation.x=0
        pose.pose.orientation.y=0
        pose.pose.orientation.z=0
        pose.pose.orientation.w=1
        predicted_trajectory.poses.append(pose)
        pose1=PoseStamped()
        pose1.pose.position.x=ref_poses[i,0]
        pose1.pose.position.y=ref_poses[i,1]
        pose1.pose.orientation.x=0
        pose1.pose.orientation.y=0
        pose1.pose.orientation.z=0
        pose1.pose.orientation.w=1
        reference_trajectory.poses.append(pose1)
        pose2=PoseStamped()
        dist = (i - N/2)*3
        angle = math.atan2(a_eqn,-b_eqn)
        pose2.pose.position.x=obstacle_points_updated[i_eqn][0] + dist*math.cos(angle)
        if b_eqn!= 0 :
            pose2.pose.position.y = (-c_eqn - a_eqn*pose2.pose.position.x)/b_eqn 
        else :
            pose2.pose.position.y = obstacle_points_updated[i_eqn][1] + dist*math.sin(angle)
        pose2.pose.orientation.x=0
        pose2.pose.orientation.y=0
        pose2.pose.orientation.z=0
        pose2.pose.orientation.w=1
        boundary_line.poses.append(pose2)
        
        # Pass nominal states and commands to communicate low level pre-compensator
        pose3=PoseStamped()
        pose3.header.stamp.secs = int((2+ff_value[i,4])*1000)
        pose3.pose.position.x=min(40, max(-40, float(buff_con[i,1]) * 180 / math.pi))
        pose3.pose.position.y = t1 + curr_time_est+curr_time_est_var*beta + T*i
        if wait_time > 0 :
            pose3.pose.position.y = t1 + wait_time+ T*i
        pose3.pose.position.z = float(buff_con[i,0])
        pose3.pose.orientation.x=ff_value[i,0]
        pose3.pose.orientation.y=ff_value[i,1]
        pose3.pose.orientation.z=ff_value[i,2]
        pose3.pose.orientation.w=ff_value[i,3]
        ctrlmsg.poses.append(pose3)

    # For plotting    
    pub3.publish(predicted_trajectory)
    pub5.publish(reference_trajectory)
    pub6.publish(boundary_line)
    print("Published ", buff_con)
    t2 = rospy.get_time()
    if wait_time < 0 :
        rospy.sleep(0.04 + 0.0*math.sin(t2/3.5))
    else :
        if (t2-t1)<wait_time :
            rospy.sleep(wait_time-(t2-t1))

    t3 = rospy.get_time()
    new_time_est = t3-t1
    pub1.publish(ctrlmsg)
    
    # Wait for remaining time to maintain consistency i.e. (upper bound time - time observed)
    if wait_time < 0 :
        if curr_time_est+curr_time_est_var*beta-new_time_est > 0 :
            rospy.sleep(curr_time_est+curr_time_est_var*beta-new_time_est)
    print("Time taken : ", new_time_est)
    # print("Commands :", buff_con)
    
    time_estimates.append([curr_time_est,new_time_est,curr_time_est+curr_time_est_var*beta,t2])
    
    # Get updated computation time estimates and variance
    curr_time_est, curr_time_est_var = update_delay_time_estimate(new_time_est)
    print(curr_time_est, curr_time_est_var)
    # p=[float(x_bot),float(y_bot),float(yaw_car),xs[0],xs[1],xs[2]]
    a=PointStamped()
    a.header.stamp=rospy.Time.now()
    a.header.frame_id='/map'
    a.point.x=xs[0]
    a.point.y=xs[1]
    a.point.z=0
    pub2.publish(a)
    print('Published at ', rospy.get_time())

# Initialization
def start():
    global pub1
    global pub2
    global pub3
    global pub4
    global pub5
    global pub6
    global inv_set
    global pub_obs 

    inv_set = get_inv_set(T,0,2,Q_robust,R_robust,N,without_steering=True)
    rospy.init_node('path_tracking', anonymous=True)
    pub1 = rospy.Publisher('cmd_delta', Path, queue_size=1)
    pub2 = rospy.Publisher('goal_point', PointStamped, queue_size=1)
    pub3 = rospy.Publisher('/predicted_trajectory', Path,queue_size=1)
    pub4 = rospy.Publisher('start_point', PointStamped, queue_size=1)
    pub5 = rospy.Publisher('/reference_trajectory', Path,queue_size=1)
    pub6 = rospy.Publisher('/constraint_line', Path,queue_size=1)
    pub_obs = rospy.Publisher('/obs_draw', PolygonStamped,queue_size=1)
    rospy.Subscriber("base_pose_ground_truth", Odometry, posCallback,queue_size=1)
    rospy.Subscriber("astroid_path", Path, pathCallback,queue_size=1)
    
    if scenario != 'static' :
        rospy.Subscriber("/obstacle/base_pose_ground_truth", Odometry, obsPosCallback,queue_size=1)
        
    rospy.Subscriber("/tf", TFMessage, callback_tf,queue_size=1)
    rospy.spin()
    r=rospy.Rate(1000)
    r.sleep()

if __name__ == '__main__':    
    start()
