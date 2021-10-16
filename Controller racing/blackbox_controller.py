# -*- coding: utf-8 -*-
from sys import path as sys_path
from os import path as os_path, times
from casadi import *
import math
import numpy as np
import rticonnextdds_connector as rti
import matplotlib.pyplot as plt
import time
import lqr_steering_adjust as lqrs
import util_funcs as utils

global x_bot
global y_bot
global control_count
control_count=0
no_of_vehicles = 0
trajectory_to_follow = []
file_path = os_path.dirname(os_path.realpath(__file__))
sys_path.append(file_path + "/../../../")
has_start=True

EPSILON = 0
L=29
opp_speed = 190*(5/18)
######### Global variables ##################

predicted_x = 0
predicted_y = 0
predicted_theta = 0
predicted_v = 0

##########     Hyperparameters     #################

gear_throttles = [2770,3320,3390,3660,3660,3800]
gear_change_speeds = [18.2,28.4,38.5,47,55.5]
air_resistance_const = 0.43
mass = 720 # in Kg
tolerance = 2
save_path_after = -1
file_centre_line = "./lap1.csv"  # File to read the global reference line, if None then centre line will be taken
file_path_follow = "./lap8.csv"  # File to read the global reference line, if None then centre line will be taken
file_new_path = "./coordinates_nc2.txt" # File in which the new coordinates will be saved
Q_along=2  # Weight for progress along the road
Q_dist=0  # Weight for distance for the center of the road
penalty_out_of_road = 6 # Penalise for planning path outside the road
no_iters = 3
max_no_of_vehicles = 4
max_vehicle_id = 10
T = .04 # Time horizon
N = 10 # Number of control intervals
kp=1 # For PID controller
ki=0
kd=0
mu = 1
g_const = 9.8
threshold = 20000
dist_threshold = 0.25
speed = 30
factor = 10
#Initialisation

curr_time_est = 0.3
curr_time_est_var = 1
var_factor = 0.001
noise_to_process_var = 1
COMPENSATE_COMP_DELAY = False 
COMPENSATE_ACTUATOR_DELAY = True

def dist1(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**(1/2)

def blackbox_controller(path,curr_pos) :
    path = np.array(path)
    target = path[-1,:]
    dx_ = (target[0] - curr_pos[0])
    dy_ = target[1] - curr_pos[1]
    dx = dx_*cos(curr_pos[2]) + dy_*sin(curr_pos[2])
    dy = dy_*cos(curr_pos[2]) - dx_*sin(curr_pos[2])

    # print("Target :", dx,dy)
    steering = 2*L*dy/(dx**2+dy**2)
    steerings = [steering]*N
    speeds = path[:,-1]
    print(steerings)
    return steerings, speeds

def path_length_distance(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def calc_path_length(x_p):
    # global path_length
    path_length = []
    for i in range(len(x_p)):
        if i == 0:
            path_length.append(0)
        else:
            path_length.append(path_length[i-1] + path_length_distance(x_p[i], x_p[i-1]))
    return path_length

def get_path(x_bot, y_bot, theta_bot, x_p, N,speed,T) :
    out_path = []
    path_lengths = calc_path_length(x_p)
    distances = []    
    for i in range(len(x_p)):
        a = x_p[i]
        distances += [path_length_distance([x_bot,y_bot],a)]
    # print(distances)
    
    ep = min(distances)
    total_index=len(x_p)
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
        x1,y1 = x_p[cp-1][0], x_p[cp-1][1]
        x2,y2 = x_p[cp][0], x_p[cp][1]
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        new_dists += [((xs-x_bot)**2 + (ys-y_bot)**2)**0.5]
    new_dists.append(ep)
    for s in range(no_of_segs_r) :
        x1,y1 = x_p[cp][0], x_p[cp][1]
        x2,y2 = x_p[cp+1][0], x_p[cp+1][1]
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        new_dists += [((xs-x_bot)**2 + (ys-y_bot)**2)**0.5]
    min_ni = new_dists.index(min(new_dists))
    if min_ni < no_of_segs_l :
        s = min_ni
        x1,y1 = x_p[cp-1][0], x_p[cp-1][1]
        x2,y2 = x_p[cp][0], x_p[cp][1]
        xs,ys = x1 + (x2-x1)*(seg_len_l/available_length_l)*(s+1), y1 + (y2-y1)*(seg_len_l/available_length_l)*(s+1)
        
        v1 = x_p[cp-1][2]
        v2 = x_p[cp][2]
        vs = v1 + (v2-v1)*(seg_len_l/available_length_l)*(s+1)
        
        # pose_temp=PoseStamped()
        # pose_temp.pose.position.x = xs
        # pose_temp.pose.position.y = ys
        x_p.insert(cp,[xs,ys,vs])
        path_lengths.insert(cp,path_lengths[cp-1] + seg_len_l*(s+1))
    if min_ni > no_of_segs_l :
        s = min_ni - no_of_segs_l - 1
        x1,y1 = x_p[cp][0], x_p[cp][1]
        x2,y2 = x_p[cp+1][0], x_p[cp+1][1]
        xs,ys = x1 + (x2-x1)*(seg_len_r/available_length_r)*(s+1), y1 + (y2-y1)*(seg_len_r/available_length_r)*(s+1)
        
        v1 = x_p[cp-1][2]
        v2 = x_p[cp][2]
        vs = v1 + (v2-v1)*(seg_len_r/available_length_r)*(s+1)
        
        x_p.insert(cp+1,[xs,ys,vs])
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
            out_path.append(np.array([x_p[k][0],x_p[k][1]]))
            continue
        a = req_dist + path_lengths[cp] - path_lengths[k-1]
        b = path_lengths[k] - req_dist - path_lengths[cp]
        X1 = np.array([x_p[k-1][0],x_p[k-1][1],x_p[k-1][2]])
        X2 = np.array([x_p[k][0],x_p[k][1],x_p[k][2]])
        X = X1*b/(a+b) + X2*a/(a+b)
        out_path.append(X)
        i = k-1
    # print(out_path)
    X1 = out_path[0]
    X2 = out_path[-1]
    x1_,y1_ = X1[0]-x_bot,X1[1]-y_bot
    x2_,y2_ = X2[0]-x_bot,X2[1]-y_bot
    x1 = x1_*cos(theta_bot) + y1_*sin(theta_bot)
    y1 = y1_*cos(theta_bot) - x1_*sin(theta_bot)
    x2 = x2_*cos(theta_bot) + y2_*sin(theta_bot)
    y2 = y2_*cos(theta_bot) - x2_*sin(theta_bot)
    dist = ((y1-y2)**2 + (x2-x1)**2)**(1/2)
    a = (y1-y2)/dist
    b = (x2-x1)/dist
    c = (y2*x1 - y1*x2)/dist
    return np.array(out_path),[a,b,c]
    
def get_future_state_est(current_pose,buff_con,curr_time_est) :
    if COMPENSATE_COMP_DELAY :
        return np.array(current_pose) + np.array([current_pose[3]*cos(current_pose[2]),current_pose[3]*sin(current_pose[2]),current_pose[3]*10*tan(buff_con[0][0]/10)/L,0])*curr_time_est/factor
    else :
        return np.array(current_pose)
    
def update_time_estimates(curr_time_est,curr_time_est_var,meas_time_est):
    curr_time_est_var = curr_time_est_var + 1
    curr_time_est = curr_time_est + (meas_time_est-curr_time_est)*noise_to_process_var/(noise_to_process_var+curr_time_est_var)
    curr_time_est_var = curr_time_est_var*noise_to_process_var/(noise_to_process_var+curr_time_est_var)
    return curr_time_est, curr_time_est_var

with rti.open_connector(
        config_name="MyParticipantLibrary::ObstacleParticipant",
        url=file_path + "/Sensors_ego1.xml") as connector:
    input_speed = connector.get_input("StateSubscriber::stateReader")
    controls = connector.get_output("controlPublisher::controlPub")
    wait_topic = connector.get_input("simWaitSub::simWaitReader")
    done_topic = connector.get_output("simDonePub::simDoneWriter")
    output = connector.get_output("SpeedPublisher::speedPub")
    output_steering = connector.get_output("steeringPublisher::steeringPub")
    input1 = connector.get_input("roadSubscriber::roadReader1")
    input2 = connector.get_input("roadSubscriber::roadReader2")
    input_radar_F = connector.get_input("radarSubscriber_F::radarReader_F")
    input_radar_left = connector.get_input("radarSubscriber_left::radarReader_left")
    input_radar_right = connector.get_input("radarSubscriber_right::radarReader_right")
    
    # Read data from the input, transform it and write it into the output
    print("Waiting for data...")
    
    #Initialise
    all_vehicles = np.ones((max_no_of_vehicles,6))*10000
    opp_vehicle_detected = np.zeros((max_no_of_vehicles),dtype = int)
    opp_vehicle_detected_state = np.zeros((max_vehicle_id),dtype = int)
    
    curr_steering = 0
    curr_speed = 0
    target_speed = 0
    aggregate = 0
    nr_dist = 0
    all_vehicles = np.ones((max_no_of_vehicles,6))*10000
    if file_path_follow != None:
        trajectory_to_follow = np.loadtxt(file_path_follow,delimiter = ",")
    else :
        trajectory_to_follow=None
    if file_centre_line != None:
        centre_line = np.loadtxt(file_centre_line,delimiter = ",")
    else :
        centre_line=None
    traj_followed = []
    itr = 0
    total_itr=0
    curr_steering_array = np.zeros((N,2))
    target_speed_array = np.array([0]*N)
    while True:
        total_itr=total_itr+1
        itr = itr+1
        if total_itr > save_path_after and save_path_after!=-1:
            break
        print("Iteration no", total_itr)
        input_speed.wait()

        wait_topic.wait()
        wait_topic.take()
        print("Waiting")
        input1.wait()
        input1.take()
        input2.wait()
        input2.take()
        input_radar_F.wait()
        input_radar_F.take()
        input_radar_left.wait()
        input_radar_left.take()
        input_radar_right.wait()
        input_radar_right.take()
        print("Received")
        wait_msg = []
        for sample in wait_topic.samples.valid_data_iter:
            data = sample.get_dictionary()
            wait_msg = data
        
        no_of_vehicles = 0
        all_vehicles[:,:2] = 10000
        all_vehicles[:,2] = 1
        all_vehicles[:,3] = 0
        all_vehicles[:,4:6] = 10000
        opp_vehicle_detected[:] = 0

        
        # Get the opponent vehicle data
        if True :
            for sample in input_radar_F.samples.valid_data_iter:
                data = sample.get_dictionary()
                
                for k in range(len(data['targetsArray'])):
                    all_vehicles[no_of_vehicles,0] = data['targetsArray'][k]['posXInChosenRef'] - 3.2
                    all_vehicles[no_of_vehicles,1] = data['targetsArray'][k]['posYInChosenRef']
                    all_vehicles[no_of_vehicles,2] = data['targetsArray'][k]['absoluteSpeedX']
                    all_vehicles[no_of_vehicles,3] = data['targetsArray'][k]['absoluteSpeedY']
                    theta_heading = data['targetsArray'][k]['posHeadingInChosenRef']
                    # all_vehicles[no_of_vehicles,2] = opp_speed*cos(theta_heading)
                    # all_vehicles[no_of_vehicles,3] = opp_speed*sin(theta_heading)
                    opp_vehicle_detected[no_of_vehicles] = data['targetsArray'][k]['scanerId']
                    all_vehicles[no_of_vehicles,0], all_vehicles[no_of_vehicles,1] \
                        = utils.anchorPointToCenter(\
                            all_vehicles[no_of_vehicles,0], \
                            all_vehicles[no_of_vehicles,1], \
                                math.atan2(all_vehicles[no_of_vehicles,3], all_vehicles[no_of_vehicles,2]),\
                                data['targetsArray'][k]['anchorPoint']) 
                    all_vehicles[no_of_vehicles,4] = all_vehicles[no_of_vehicles,0] + 1.791102
                    all_vehicles[no_of_vehicles,5] = all_vehicles[no_of_vehicles,1]

                    print("Vehicle no ", no_of_vehicles)
                    print("Vehicle id : ",opp_vehicle_detected[no_of_vehicles])
                    print("X : ", all_vehicles[no_of_vehicles,0])
                    print("Y : ", all_vehicles[no_of_vehicles,1])
                    print("Ego Vehicle frame X : ", all_vehicles[no_of_vehicles,4])
                    print("Ego Vehicle frame Y : ", all_vehicles[no_of_vehicles,5])
                    print("Speed X : ", all_vehicles[no_of_vehicles,2])
                    print("Speed Y : ", all_vehicles[no_of_vehicles,3])
                    print("detectionStatus :", data['targetsArray'][k]['detectionStatus'])
                    print("type :", data['targetsArray'][k]['type_'])
                    #print("name :", data['targetsArray'][k]['name'])
                    print("beamIndex :", data['targetsArray'][k]['beamIndex'])
                    print("existenceTime :", data['targetsArray'][k]['existenceTime'])
                    print("anchorPoint :", data['targetsArray'][k]['anchorPoint'])
                    print("referenceFrame :", data['targetsArray'][k]['referenceFrame'])
                    no_of_vehicles += 1
                break
            print("From left radar")
            for sample in input_radar_left.samples.valid_data_iter:
                data = sample.get_dictionary()
                for k in range(len(data['targetsArray'])):
                    if(data['targetsArray'][k]['posYInChosenRef']>-2 or data['targetsArray'][k]['posXInChosenRef']<0 or data['targetsArray'][k]['posXInChosenRef']>5) :
                        continue
                    all_vehicles[no_of_vehicles,0] = -data['targetsArray'][k]['posYInChosenRef']
                    all_vehicles[no_of_vehicles,1] = data['targetsArray'][k]['posXInChosenRef']
                    all_vehicles[no_of_vehicles,2] = -data['targetsArray'][k]['absoluteSpeedY']
                    all_vehicles[no_of_vehicles,3] = data['targetsArray'][k]['absoluteSpeedX']
                    opp_vehicle_detected[no_of_vehicles] = data['targetsArray'][k]['scanerId']
                    all_vehicles[no_of_vehicles,0], all_vehicles[no_of_vehicles,1] \
                        = utils.anchorPointToCenter(\
                            all_vehicles[no_of_vehicles,0], \
                            all_vehicles[no_of_vehicles,1], \
                                math.atan2(all_vehicles[no_of_vehicles,3], all_vehicles[no_of_vehicles,2]),\
                                data['targetsArray'][k]['anchorPoint']) 
                    all_vehicles[no_of_vehicles,4] = all_vehicles[no_of_vehicles,0] + 0.220795
                    all_vehicles[no_of_vehicles,5] = all_vehicles[no_of_vehicles,1] + 2.064369
                    print("Vehicle no ", no_of_vehicles)
                    print("Vehicle id : ",opp_vehicle_detected[no_of_vehicles])
                    print("X : ", all_vehicles[no_of_vehicles,0])
                    print("Y : ", all_vehicles[no_of_vehicles,1])
                    print("Ego Vehicle frame X : ", all_vehicles[no_of_vehicles,4])
                    print("Ego Vehicle frame Y : ", all_vehicles[no_of_vehicles,5])
                    print("Speed X : ", all_vehicles[no_of_vehicles,2])
                    print("Speed Y : ", all_vehicles[no_of_vehicles,3])
                    print("detectionStatus :", data['targetsArray'][k]['detectionStatus'])
                    print("type :", data['targetsArray'][k]['type_'])
                    #print("name :", data['targetsArray'][k]['name'])
                    print("beamIndex :", data['targetsArray'][k]['beamIndex'])
                    print("existenceTime :", data['targetsArray'][k]['existenceTime'])
                    print("anchorPoint :", data['targetsArray'][k]['anchorPoint'])
                    print("referenceFrame :", data['targetsArray'][k]['referenceFrame'])
                    no_of_vehicles +=1
                break
            
            print("From right radar")
            for sample in input_radar_right.samples.valid_data_iter:
                data = sample.get_dictionary()
                for k in range(len(data['targetsArray'])):
                    if (data['targetsArray'][k]['posXInChosenRef']<0 or data['targetsArray'][k]['posXInChosenRef']>5) :
                        continue
                    all_vehicles[no_of_vehicles,0] = data['targetsArray'][k]['posYInChosenRef']
                    all_vehicles[no_of_vehicles,1] = -data['targetsArray'][k]['posXInChosenRef']
                    all_vehicles[no_of_vehicles,2] = data['targetsArray'][k]['absoluteSpeedY']
                    all_vehicles[no_of_vehicles,3] = -data['targetsArray'][k]['absoluteSpeedX']
                    opp_vehicle_detected[no_of_vehicles] = data['targetsArray'][k]['scanerId']
                    all_vehicles[no_of_vehicles,0], all_vehicles[no_of_vehicles,1] \
                        = utils.anchorPointToCenter(\
                            all_vehicles[no_of_vehicles,0], \
                            all_vehicles[no_of_vehicles,1], \
                                math.atan2(all_vehicles[no_of_vehicles,3], all_vehicles[no_of_vehicles,2]),\
                                data['targetsArray'][k]['anchorPoint']) 
                    all_vehicles[no_of_vehicles,4] = all_vehicles[no_of_vehicles,0] - 0.220795
                    all_vehicles[no_of_vehicles,5] = all_vehicles[no_of_vehicles,1] + 2.064369 
                    print("Vehicle no ", no_of_vehicles)
                    print("Vehicle id : ",opp_vehicle_detected[no_of_vehicles])
                    print("X : ", all_vehicles[no_of_vehicles,0])
                    print("Y : ", all_vehicles[no_of_vehicles,1])
                    print("Ego Vehicle frame X : ", all_vehicles[no_of_vehicles,4])
                    print("Ego Vehicle frame Y : ", all_vehicles[no_of_vehicles,5])
                    print("Speed X : ", all_vehicles[no_of_vehicles,2])
                    print("Speed Y : ", all_vehicles[no_of_vehicles,3])
                    print("detectionStatus :", data['targetsArray'][k]['detectionStatus'])
                    print("type :", data['targetsArray'][k]['type_'])
                    #print("name :", data['targetsArray'][k]['name'])
                    print("beamIndex :", data['targetsArray'][k]['beamIndex'])
                    print("existenceTime :", data['targetsArray'][k]['existenceTime'])
                    print("anchorPoint :", data['targetsArray'][k]['anchorPoint'])
                    print("referenceFrame :", data['targetsArray'][k]['referenceFrame'])
                    no_of_vehicles += 1
                break
        
        px = 0
        py = 0
        angle_heading = 0
        input_speed.take()
        for sample in input_speed.samples.valid_data_iter:
            t1 = time.time()
            data = sample.get_dictionary()
            vx = data['cdgSpeed_x']
            vy = data['cdgSpeed_y']
            vz = data['cdgSpeed_z']
            px = data['cdgPos_x']  
            steering_ = data['cdgSpeed_heading']
            
            py = data['cdgPos_y']  
            angle_heading = data['cdgPos_heading']
            curr_speed = math.sqrt(vx*vx+vy*vy+vz*vz)
            # print("Current State :",[px,py,angle_heading,curr_speed])
            # print("Predicted State :",[predicted_x,predicted_y,predicted_theta,predicted_v])
            if itr>10 and save_path_after!=-1:
                itr = 0
                traj_followed.append([px,py,curr_speed])
            # print("Current Speed : ", curr_speed)
            # break
            curr_pos = [px,py,angle_heading,curr_speed]
            curr_pos = get_future_state_est(curr_pos,[curr_steering_array[:,1],np.array([25]*N)],curr_time_est)
            path_array,_ = get_path(px,py,angle_heading,trajectory_to_follow.tolist(),int(2*N),speed,T)
            _,centre_line_eq = get_path(px,py,angle_heading,centre_line.tolist(),int(2*N),speed,T)
            curr_steering_array, target_speed_array = blackbox_controller(path_array,curr_pos)
            curr_steering = steering_*29/80
            print("Waiting here")
            if COMPENSATE_ACTUATOR_DELAY :
                curr_steering_array = np.array(lqrs.get_optimal_commands(all_vehicles, curr_steering_array,curr_steering,curr_pos[-1],centre_line_eq))
            print("Finished here")
            curr_steering = float(curr_steering_array[0,1])
            target_speed = float(curr_steering_array[0,0])
            out_controls = {}
            S1 = np.zeros(N).astype(float)
            S2 = np.zeros(N).astype(float)
            for i in range(N):
                S1[i] = curr_steering_array[i,0]
                S2[i] = curr_steering_array[i,1]
            # print(S1)
            # print(S2)
            out_controls['speedsArray'] = S1.tolist()
            out_controls['steeringArray'] = S2.tolist()
            # time.sleep(0.03)
            # time.sleep(0.3)
            t2 = time.time()
            new_time_est = t2-t1
            if curr_time_est+2*curr_time_est_var*var_factor-new_time_est > 0 :
                time.sleep(curr_time_est+2*curr_time_est_var*var_factor-new_time_est)
            print("Time taken : ", new_time_est)
            curr_time_est, curr_time_est_var = update_time_estimates(curr_time_est,curr_time_est_var,new_time_est)
    
            print("Time taken : ", curr_time_est)
            controls.instance.set_dictionary(out_controls)
            controls.write()
            break
        
        print("Time: ", data['TimeOfUpdate'])
        print("message written")
    
    traj_followed = np.array(traj_followed).T/100.0
    print("Trajectory followed :-")
    print(traj_followed)
    plt.plot(traj_followed[0],traj_followed[1],'k', lw=0.5, alpha=0.5)
    plt.plot(trajectory_to_follow[0]/100.0,trajectory_to_follow[1]/100.0,'--k', lw=0.5, alpha=0.5)
    np.savetxt(file_new_path, traj_followed, delimiter=',')
    plt.show()

