# -*- coding: utf-8 -*-
from sys import path as sys_path
from os import path as os_path, times
from casadi import *
import math
import numpy as np
import rticonnextdds_connector as rti
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle 
import time

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
save_path_after = 100
file_path_follow = "./lap1.csv"  # File to read the global reference line, if None then centre line will be taken
file_new_path = "./steerings_32.txt" # File in which the new coordinates will be saved
file_prev_path = "./steerings_16.txt" # File in which the new coordinates will be saved
Q_along=2  # Weight for progress along the road
Q_dist=0  # Weight for distance for the center of the road
penalty_out_of_road = 6 # Penalise for planning path outside the road
no_iters = 3
max_no_of_vehicles = 4
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
curr_time_est = 0
factor = 10
#Initialisation

steerings = []

with rti.open_connector(
        config_name="MyParticipantLibrary::ObstacleParticipant",
        url=file_path + "/../Sensors_ego1.xml") as connector:
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
    curr_steering = 0
    curr_speed = 0
    target_speed = 0
    aggregate = 0
    nr_dist = 0
    all_vehicles = np.ones((max_no_of_vehicles,4))*10000
    if file_path_follow != None:
        trajectory_to_follow = np.loadtxt(file_path_follow,delimiter = ",")
    else :
        trajectory_to_follow=None
    traj_followed = []
    itr = 0
    total_itr=0
    curr_steering_array = np.array([0]*N)
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
        # print("Received")
        wait_msg = []
        for sample in wait_topic.samples.valid_data_iter:
            data = sample.get_dictionary()
            wait_msg = data
        
        px = 0
        py = 0
        angle_heading = 0
        input_speed.take()
        for sample in input_speed.samples.valid_data_iter:
            t1 = time.time()
            data = sample.get_dictionary()
            steering = data['SteeringWheelAngle']
            steering_speed = data['cdgSpeed_heading']
            steering_torque = data['SteeringWheelTorque']
            vx = data['cdgSpeed_x']
            vy = data['cdgSpeed_y']
            vz = data['cdgSpeed_z']
            px = data['cdgPos_x']
            py = data['cdgPos_y']
            angle_heading = data['cdgPos_heading']
            steerings.append([data['TimeOfUpdate'],steering,steering_speed,steering_torque])
            curr_speed = math.sqrt(vx*vx+vy*vy+vz*vz)
            # print("Current State :",[px,py,angle_heading,curr_speed])
            # print("Predicted State :",[predicted_x,predicted_y,predicted_theta,predicted_v])
            if itr>10 and save_path_after!=-1:
                itr = 0
                traj_followed.append([px,py,curr_speed])
            # print("Current Speed : ", curr_speed)
            # break
            break
        out = {}
        target_speed=1
        #print(target_speed-curr_speed)
        out['AcceleratorAdditive'] = 1
        out['AcceleratorMultiplicative'] = 0
        out['BrakeAdditive'] = 0
        out['BrakeMultiplicative'] = 0
        out['ClutchAdditive'] = 0
        out['ClutchMultiplicative'] = 0
        out['GearboxAutoMode'] = 9
        out['GearboxTakeOver'] = 1
        out['IsRatioLimit'] = 0
        out['MaxRatio'] = 1000
        out['MinRatio'] = 1
        out['ParkingBrakeAdditive'] = 0
        out['ParkingBrakeMultiplicative'] = 0
        out['ShiftDown'] = 0
        out['ShiftUp'] = 0
        out['WantedGear'] = 1
        
        out['TimeOfUpdate'] = data['TimeOfUpdate']
        # if curr_index==0:
        output.instance.set_dictionary(out)
        output.write()
        # print("Current Index:", curr_index)
        # print("Target Speed:", target_speed)
        out_steering = {}
        
        curr_steering = curr_steering_array[0]
        out_steering['AdditiveSteeringWheelAngle'] = math.pi/30         
        out_steering['AdditiveSteeringWheelAccel'] = 0
        out_steering['AdditiveSteeringWheelSpeed'] = 0
        out_steering['AdditiveSteeringWheelTorque'] = 0
        out_steering['MultiplicativeSteeringWheelAccel'] = 1
        out_steering['MultiplicativeSteeringWheelAngle'] = 0
        out_steering['MultiplicativeSteeringWheelSpeed'] = 1
        out_steering['MultiplicativeSteeringWheelTorque'] = 1
        out_steering['TimeOfUpdate'] = data['TimeOfUpdate']
        print("Steering Command : " , curr_steering)
        # if curr_index==0:
        output_steering.instance.set_dictionary(out_steering)
        output_steering.write()
        print("Time: ", data['TimeOfUpdate'])
        # print(len(wait_msg), wait_msg)
        done_topic.instance.set_dictionary(wait_msg)
        done_topic.write()
        print("message written")
    
    # traj_followed = np.array(traj_followed).T/100.0
    print("Steerings :-")
    print(steerings)
    steerings = np.array(steerings)
    plt.plot(steerings[:,0],steerings[:,2], lw=0.5)
    plt.plot(steerings[:,0],0.13*(1-np.exp(-20*steerings[:,0])),'k', lw=0.5, alpha=0.5)
    # plt.plot(trajectory_to_follow[0]/100.0,trajectory_to_follow[1]/100.0,'--k', lw=0.5, alpha=0.5)
    np.savetxt(file_new_path, steerings, delimiter=',')
    steerings = np.loadtxt(file_prev_path, delimiter=',')
    plt.plot(steerings[:,0],steerings[:,2], lw=0.5)
    plt.plot(steerings[:,0],0.13*(1-np.exp(-20*steerings[:,0])),'k', lw=0.5, alpha=0.5)
    
    plt.show()

